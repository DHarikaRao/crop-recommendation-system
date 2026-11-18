"""Flask website and JSON API for the crop recommendation model."""
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from flask import Flask, Response, abort, g, jsonify, render_template, request, send_from_directory

from crop_model import (
    FEATURE_INFO,
    FEATURES,
    confidence_level,
    crop_icon,
    display_name,
    explain,
    load_artifact,
    ood_check,
    plausible_crops,
    predict,
    range_warnings,
    validate_input,
)
from metrics import Metrics
from openapi import build_spec
from ratelimit import RateLimiter
from viz import diverging_bars, radar, range_marker

BASE_DIR = Path(__file__).resolve().parent
MAX_BATCH = 100
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def create_app(model_path=None, reports_dir=None, rate_limit=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # inputs are tiny; reject anything larger
    app.logger.setLevel(logging.INFO)
    app.jinja_env.globals["crop_icon"] = crop_icon

    if rate_limit is None:
        rate_limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
    limiter = RateLimiter(limit=rate_limit, window=60.0)
    metrics = Metrics()

    model_path = Path(model_path or os.environ.get("MODEL_PATH") or BASE_DIR / "models" / "crop_model.joblib")
    reports_dir = Path(reports_dir or BASE_DIR / "reports")
    artifact = load_artifact(model_path) if model_path.exists() else None
    if artifact:
        app.logger.info("Loaded %s (%d crops)", artifact["metrics"]["best_model"], len(artifact["classes"]))
    else:
        app.logger.warning("No model at %s. Run train.py first.", model_path)

    # ------------------------------------------------------------ middleware
    @app.before_request
    def start_request():
        g.started = time.perf_counter()
        incoming = request.headers.get("X-Request-ID", "")
        g.request_id = incoming if REQUEST_ID_PATTERN.match(incoming) else uuid.uuid4().hex[:12]

    @app.before_request
    def enforce_rate_limit():
        if request.method != "POST":
            return None
        allowed, retry_after = limiter.check(request.remote_addr or "unknown")
        if allowed:
            return None
        response = jsonify(error="Rate limit exceeded. Try again shortly.", retry_after_seconds=retry_after)
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response

    @app.after_request
    def record_metrics(response):
        request_id = getattr(g, "request_id", "-")
        response.headers["X-Request-ID"] = request_id
        started = getattr(g, "started", None)
        if started is not None:
            elapsed = time.perf_counter() - started
            endpoint = request.url_rule.rule if request.url_rule else "unmatched"  # bounded label set
            metrics.observe_request(request.method, endpoint, response.status_code, elapsed)
            app.logger.info(json.dumps({
                "request_id": request_id, "method": request.method, "path": request.path,
                "status": response.status_code, "duration_ms": round(elapsed * 1000, 2),
            }))
        return response

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'"
        )
        return response

    # --------------------------------------------------------------- helpers
    def page_context(**extra):
        # Start from a real crop's typical conditions so the first answer is meaningful.
        default_crop = None
        if artifact:
            default_crop = "rice" if "rice" in artifact["profiles"] else artifact["classes"][0]
        fields, options = [], []
        if artifact:
            options = [(c, display_name(c), crop_icon(c)) for c in artifact["classes"]]
        for name in FEATURES:
            info = FEATURE_INFO[name]
            field = {"name": name, **info}
            if artifact:
                low, high = artifact["ranges"][name]
                field.update(train_low=round(low, 1), train_high=round(high, 1), marker=range_marker(low, high, info["min"], info["max"]))
            fields.append(field)
        return {
            "fields": fields,
            "profile_options": options,
            "profiles": artifact["profiles"] if artifact else {},
            "defaults": artifact["profiles"][default_crop] if artifact else {},
            "default_crop": default_crop,
            "conformal": artifact["metrics"]["conformal"] if artifact else None,
            "stats": artifact["metrics"] if artifact else None,
            "model_loaded": artifact is not None,
            **extra,
        }

    def recommend(values):
        top = predict(artifact, values, top_k=3)
        best = top[0]
        metrics.count_prediction(best["crop"])
        warnings = range_warnings(artifact, values)
        ood = ood_check(artifact, values)
        unusual = bool(ood and ood["flagged"])
        if unusual:
            warnings.append(
                "This combination of inputs is unusual compared with the training data, so treat the result with extra caution."
            )
        clears_bar = best["probability"] >= 1.0 - artifact["conformal"]["qhat"] - 1e-12
        explanation = explain(artifact, values, best["crop"])
        for row, bar in zip(explanation, diverging_bars([r["effect"] for r in explanation])):
            row["bar_x"], row["bar_w"] = bar["x"], bar["w"]
        labels = {f: FEATURE_INFO[f]["label"].split(" (")[0] for f in FEATURES}
        return {
            "top": {**best, "icon": crop_icon(best["crop"]), "percent": round(best["probability"] * 100)},
            "others": [{**r, "icon": crop_icon(r["crop"]), "percent": round(r["probability"] * 100)} for r in top[1:]],
            "level": confidence_level(best["probability"]),
            "donut_offset": round(100 - best["probability"] * 100, 1),
            "plausible": [{**r, "icon": crop_icon(r["crop"])} for r in plausible_crops(artifact, values)],
            "coverage_target": artifact["metrics"]["conformal"]["target_coverage"],
            "unusual_input": unusual,
            "uncertain": not clears_bar,
            "warnings": warnings,
            "explanation": explanation,
            "radar": radar(FEATURES, labels, artifact["ranges"], values, artifact["profiles"][best["crop"]]),
        }

    def serialize(result):
        def crop_row(r):
            return {"crop": r["crop"], "name": r["name"], "probability": round(r["probability"], 4)}

        return {
            "recommendations": [crop_row(r) for r in [result["top"], *result["others"]]],
            "confidence": result["level"],
            "plausible_crops": [crop_row(r) for r in result["plausible"]],
            "coverage_target": result["coverage_target"],
            "unusual_input": result["unusual_input"],
            "uncertain": result["uncertain"],
            "warnings": result["warnings"],
            "explanation": [
                {
                    "feature": e["feature"], "value": e["value"], "typical": round(e["typical"], 2),
                    "effect_pp": round(e["effect"] * 100, 2), "direction": e["direction"],
                }
                for e in result["explanation"]
            ],
        }

    # ---------------------------------------------------------------- pages
    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method in ("GET", "HEAD"):
            return render_template("index.html", **page_context(form={}, errors={}, result=None))
        if artifact is None:
            return render_template("index.html", **page_context(form={}, errors={}, result=None)), 503
        values, errors = validate_input(request.form)
        if errors:
            return render_template(
                "index.html", **page_context(form=request.form, errors=errors, result=None)
            ), 400
        return render_template(
            "index.html", **page_context(form=request.form, errors={}, result=recommend(values))
        )

    @app.get("/about")
    def about():
        return render_template(
            "about.html",
            metrics=artifact["metrics"] if artifact else None,
            has_confusion=(reports_dir / "confusion_matrix.png").exists(),
            has_importance=(reports_dir / "feature_importance.png").exists(),
            has_robustness=(reports_dir / "robustness.png").exists(),
            has_calibration=(reports_dir / "calibration.png").exists(),
            model_loaded=artifact is not None,
        )

    @app.get("/reports/<path:filename>")
    def report_file(filename):
        if not filename.endswith(".png"):
            abort(404)
        return send_from_directory(reports_dir, filename)

    # ------------------------------------------------------------------ API
    @app.post("/api/predict")
    def api_predict():
        if artifact is None:
            return jsonify(error="Model not available. Run train.py first."), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send a JSON object with the seven input fields."), 400
        values, errors = validate_input(payload)
        if errors:
            return jsonify(errors=errors), 400
        return jsonify(serialize(recommend(values)))

    @app.post("/api/predict/batch")
    def api_predict_batch():
        if artifact is None:
            return jsonify(error="Model not available. Run train.py first."), 503
        payload = request.get_json(silent=True)
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_BATCH:
            return jsonify(error=f'Send {{"items": [...]}} with 1 to {MAX_BATCH} objects.'), 400
        results = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                results.append({"index": index, "errors": {"item": "Must be an object."}})
                continue
            values, errors = validate_input(item)
            results.append({"index": index, "errors": errors} if errors else {"index": index, **serialize(recommend(values))})
        return jsonify(results=results)

    @app.get("/api/model")
    def api_model():
        if artifact is None:
            return jsonify(error="Model not available. Run train.py first."), 503
        m = artifact["metrics"]
        return jsonify(
            model=m["best_model"],
            crops=artifact["classes"],
            features=artifact["features"],
            dataset_rows=m["dataset_rows"],
            test_accuracy=round(m["test_accuracy"], 4),
            test_macro_f1=round(m["test_macro_f1"], 4),
            calibration_error=round(m["calibration"]["ece"], 4),
            log_loss=round(m["calibration"]["log_loss"], 4),
            conformal_coverage=round(m["conformal"]["empirical_coverage"], 4),
            mean_set_size=round(m["conformal"]["mean_set_size"], 3),
        )

    @app.get("/openapi.json")
    def openapi_spec():
        return jsonify(build_spec())

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(
            metrics.render(model_loaded=artifact is not None),
            mimetype="text/plain",
            headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
        )

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=artifact is not None)

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(413)
    def too_large(_):
        return render_template("error.html", code=413, message="Request too large."), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False)
