"""Tests use a small synthetic dataset, so they run in seconds and need no downloads.

Run with:  python -m pytest -q      (or)      python -m unittest discover -s tests
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402
from metrics import Metrics  # noqa: E402
from openapi import build_spec  # noqa: E402
from ratelimit import RateLimiter  # noqa: E402
from viz import diverging_bars, radar, range_marker  # noqa: E402
from crop_model import (  # noqa: E402
    FEATURES,
    confidence_level,
    explain,
    load_dataset,
    ood_check,
    plausible_crops,
    predict,
    prediction_sets,
    range_warnings,
    train_and_evaluate,
    validate_input,
)

VALID = {"N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82, "ph": 6.5, "rainfall": 200}


def synthetic_dataset(per_class=40, seed=0):
    """Four fake crops with clearly different average conditions."""
    rng = np.random.default_rng(seed)
    centers = {
        "alpha": [90, 40, 40, 21, 82, 6.5, 200],
        "beta": [20, 60, 80, 28, 50, 7.0, 90],
        "gamma": [110, 20, 30, 33, 65, 5.8, 120],
        "delta": [40, 120, 200, 18, 90, 6.2, 150],
    }
    rows = []
    for label, center in centers.items():
        noise = rng.normal(0, [5, 5, 5, 1, 3, 0.2, 10], size=(per_class, 7))
        for row in np.array(center) + noise:
            rows.append([*row, label])
    return pd.DataFrame(rows, columns=FEATURES + ["label"])


class TrainedModelCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        csv = cls.root / "crops.csv"
        synthetic_dataset().to_csv(csv, index=False)
        cls.df = load_dataset(csv)
        small = {
            "Decision Tree": make_pipeline(DecisionTreeClassifier(random_state=0)),
            "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=20, random_state=0)),
        }
        cls.artifact = train_and_evaluate(
            cls.df, cls.root / "models", cls.root / "reports", cv_folds=3, candidates=small
        )
        cls.app = create_app(cls.root / "models" / "crop_model.joblib", cls.root / "reports", rate_limit=0)
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestTraining(TrainedModelCase):
    def test_outputs_are_written(self):
        for name in ["models/crop_model.joblib", "reports/metrics.json",
                     "reports/model_comparison.md", "reports/confusion_matrix.png",
                     "reports/feature_importance.png"]:
            self.assertTrue((self.root / name).exists(), name)

    def test_metrics_are_consistent(self):
        metrics = json.loads((self.root / "reports" / "metrics.json").read_text())
        self.assertEqual(metrics["n_classes"], 4)
        self.assertEqual(
            metrics["train_rows"] + metrics["calibration_rows"] + metrics["test_rows"], metrics["dataset_rows"]
        )
        self.assertGreater(metrics["test_accuracy"], 0.9)  # synthetic classes are well separated

    def test_best_model_is_chosen_by_cross_validation(self):
        metrics = self.artifact["metrics"]
        best_cv = max(r["cv_accuracy_mean"] for r in metrics["comparison"])
        chosen = next(r for r in metrics["comparison"] if r["model"] == metrics["best_model"])
        self.assertEqual(chosen["cv_accuracy_mean"], best_cv)

    def test_prediction_returns_sorted_probabilities(self):
        values, errors = validate_input(VALID)
        self.assertEqual(errors, {})
        result = predict(self.artifact, values, top_k=3)
        probs = [r["probability"] for r in result]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertLessEqual(sum(probs), 1.0 + 1e-9)
        self.assertEqual(result[0]["crop"], "alpha")

    def test_extra_reports_are_written(self):
        for name in ["reports/model_card.md", "reports/robustness.png", "reports/calibration.png"]:
            self.assertTrue((self.root / name).exists(), name)
        card = (self.root / "reports" / "model_card.md").read_text(encoding="utf-8")
        self.assertIn("Model card", card)
        self.assertIn("Calibration", card)

    def test_robustness_is_measured_and_noise_hurts(self):
        robustness = self.artifact["metrics"]["robustness"]
        clean = robustness["noise"][0]["accuracy"]
        self.assertEqual(clean, self.artifact["metrics"]["test_accuracy"])  # level 0 == untouched test set
        self.assertLessEqual(robustness["noise"][-1]["accuracy"], clean)
        self.assertLessEqual(robustness["missing"][-1]["accuracy"], robustness["missing"][0]["accuracy"])

    def test_calibration_report(self):
        calibration = self.artifact["metrics"]["calibration"]
        self.assertGreaterEqual(calibration["ece"], 0.0)
        self.assertLessEqual(calibration["ece"], 1.0)
        self.assertGreater(calibration["log_loss"], 0.0)
        self.assertEqual(sum(b["count"] for b in calibration["bins"]), self.artifact["metrics"]["test_rows"])

    def test_explanation_covers_every_input_and_is_sorted(self):
        values, _ = validate_input(VALID)
        rows = explain(self.artifact, values, "alpha")
        self.assertEqual({r["feature"] for r in rows}, set(FEATURES))
        effects = [abs(r["effect"]) for r in rows]
        self.assertEqual(effects, sorted(effects, reverse=True))

    def test_explanation_is_zero_when_inputs_equal_the_baseline(self):
        baseline = dict(self.artifact["baseline"])
        rows = explain(self.artifact, baseline, "alpha")
        self.assertTrue(all(abs(r["effect"]) < 1e-12 for r in rows))
        self.assertTrue(all(r["direction"] == "neutral" for r in rows))

    def test_conformal_sets_cover_and_include_the_top_crop(self):
        conformal = self.artifact["metrics"]["conformal"]
        self.assertEqual(conformal["target_coverage"], 0.9)
        self.assertGreaterEqual(conformal["empirical_coverage"], 0.8)
        self.assertGreaterEqual(conformal["mean_set_size"], 1.0)
        self.assertGreaterEqual(conformal["qhat"], 0.0)
        self.assertLessEqual(conformal["qhat"], 1.0)
        values, _ = validate_input(VALID)
        plausible = plausible_crops(self.artifact, values)
        self.assertEqual(plausible[0]["crop"], predict(self.artifact, values)[0]["crop"])

    def test_prediction_sets_keep_confident_crops_and_fall_back_when_unsure(self):
        proba = np.array([[0.90, 0.05, 0.03, 0.02], [0.40, 0.30, 0.20, 0.10]])
        mask = prediction_sets(proba, qhat=0.2)  # bar = 0.8
        self.assertEqual(np.flatnonzero(mask[0]).tolist(), [0])        # clears the bar: single crop
        self.assertEqual(np.flatnonzero(mask[1]).tolist(), [0, 1, 2])  # nothing clears it: three best guesses
        self.assertEqual(prediction_sets(proba, qhat=1.0).sum(), 8)    # bar = 0: every class

    def test_uncertain_flag_and_fallback_fraction(self):
        conformal = self.artifact["metrics"]["conformal"]
        self.assertGreaterEqual(conformal["fallback_fraction"], 0.0)
        self.assertLessEqual(conformal["fallback_fraction"], 1.0)

    def test_unusual_inputs_are_flagged_and_typical_ones_are_not(self):
        typical, _ = validate_input(VALID)
        extreme, _ = validate_input(dict(N=0, P=300, K=400, temperature=60, humidity=0, ph=14, rainfall=500))
        self.assertFalse(ood_check(self.artifact, typical)["flagged"])
        self.assertTrue(ood_check(self.artifact, extreme)["flagged"])

    def test_search_estimators_work_end_to_end(self):
        """Tuned models (RandomizedSearchCV) must survive CV, permutation importance and joblib."""
        tuned = {
            "Tuned Tree": RandomizedSearchCV(
                make_pipeline(DecisionTreeClassifier(random_state=0)),
                {"decisiontreeclassifier__max_depth": [3, 5, None]},
                n_iter=2, cv=2, random_state=0,
            )
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifact = train_and_evaluate(
                self.df, Path(tmp) / "m", Path(tmp) / "r", cv_folds=2, candidates=tuned
            )
            self.assertIn("max_depth", artifact["metrics"]["comparison"][0]["best_params"])
            self.assertTrue((Path(tmp) / "m" / "crop_model.joblib").exists())

    def test_out_of_range_input_gives_warning(self):
        values = dict(VALID, rainfall=450)
        self.assertTrue(range_warnings(self.artifact, values))
        self.assertEqual(range_warnings(self.artifact, VALID), [])


class TestValidation(unittest.TestCase):
    def test_valid_input(self):
        values, errors = validate_input({k: str(v) for k, v in VALID.items()})
        self.assertEqual(errors, {})
        self.assertEqual(values["ph"], 6.5)

    def test_missing_and_non_numeric(self):
        _, errors = validate_input({"N": "abc", "P": ""})
        self.assertIn("N", errors)
        self.assertIn("P", errors)
        self.assertIn("rainfall", errors)

    def test_out_of_bounds_and_special_values(self):
        _, errors = validate_input(dict(VALID, ph=15, humidity="nan", N=-1, K=True))
        self.assertEqual(set(errors), {"ph", "humidity", "N", "K"})

    def test_confidence_levels(self):
        self.assertEqual(confidence_level(0.9), "High")
        self.assertEqual(confidence_level(0.6), "Medium")
        self.assertEqual(confidence_level(0.2), "Low")


class TestWebApp(TrainedModelCase):
    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Recommend a crop", response.data)

    def test_form_prediction(self):
        response = self.client.post("/", data={k: str(v) for k, v in VALID.items()})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Recommended crop", response.data)
        self.assertIn(b"Alpha", response.data)

    def test_form_validation_errors(self):
        response = self.client.post("/", data=dict({k: str(v) for k, v in VALID.items()}, ph="99"))
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"must be between", response.data)

    def test_api_prediction(self):
        response = self.client.post("/api/predict", json=VALID)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["recommendations"][0]["crop"], "alpha")
        self.assertIn(body["confidence"], {"High", "Medium", "Low"})

    def test_api_reports_whether_the_answer_is_uncertain(self):
        body = self.client.post("/api/predict", json=VALID).get_json()
        self.assertIn(body["uncertain"], (True, False))
        if body["uncertain"]:
            self.assertGreaterEqual(len(body["plausible_crops"]), 2)

    def test_form_starts_from_a_real_crops_typical_conditions(self):
        html = self.client.get("/").get_data(as_text=True)
        profile = self.artifact["profiles"]["alpha"]  # synthetic data has no "rice", so the first crop is used
        self.assertIn(f'value="{profile["N"]}"', html)
        self.assertIn('data-crop="alpha" aria-pressed="true"', html)

    def test_api_includes_explanation(self):
        body = self.client.post("/api/predict", json=VALID).get_json()
        self.assertEqual(len(body["explanation"]), len(FEATURES))
        self.assertIn(body["explanation"][0]["direction"], {"supports", "lowers", "neutral"})

    def test_model_info_endpoint(self):
        body = self.client.get("/api/model").get_json()
        self.assertEqual(body["model"], self.artifact["metrics"]["best_model"])
        self.assertEqual(sorted(body["crops"]), sorted(["alpha", "beta", "gamma", "delta"]))

    def test_result_page_shows_explanation(self):
        response = self.client.post("/", data={k: str(v) for k, v in VALID.items()})
        self.assertIn(b"Why this crop?", response.data)

    def test_api_errors(self):
        self.assertEqual(self.client.post("/api/predict", json={"N": 1}).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", data="not json").status_code, 400)
        self.assertEqual(self.client.post("/api/predict", json=[1, 2]).status_code, 400)

    def test_other_pages_and_security_headers(self):
        self.assertEqual(self.client.get("/about").status_code, 200)
        self.assertEqual(self.client.get("/health").get_json()["model_loaded"], True)
        self.assertEqual(self.client.get("/nope").status_code, 404)
        self.assertEqual(self.client.get("/reports/metrics.json").status_code, 404)
        headers = self.client.get("/").headers
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("Content-Security-Policy", headers)

    def test_metrics_endpoint_reports_traffic_and_predictions(self):
        self.client.post("/api/predict", json=VALID)
        self.client.get("/nope")
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["Content-Type"])
        text = response.get_data(as_text=True)
        self.assertIn('crop_http_requests_total{method="POST",endpoint="/api/predict",status="200"}', text)
        self.assertIn('endpoint="unmatched",status="404"', text)  # unknown paths do not create new label values
        self.assertIn('crop_predictions_total{crop="alpha"}', text)
        self.assertIn("crop_model_loaded 1", text)

    def test_api_includes_plausible_set_and_unusual_flag(self):
        body = self.client.post("/api/predict", json=VALID).get_json()
        self.assertEqual(body["plausible_crops"][0]["crop"], body["recommendations"][0]["crop"])
        self.assertFalse(body["unusual_input"])
        extreme = dict(N=0, P=300, K=400, temperature=60, humidity=0, ph=14, rainfall=500)
        flagged = self.client.post("/api/predict", json=extreme).get_json()
        self.assertTrue(flagged["unusual_input"])
        self.assertTrue(any("unusual" in w for w in flagged["warnings"]))

    def test_batch_endpoint_returns_one_result_per_item(self):
        response = self.client.post("/api/predict/batch", json={"items": [VALID, {"N": 1}, "oops"]})
        self.assertEqual(response.status_code, 200)
        results = response.get_json()["results"]
        self.assertEqual([r["index"] for r in results], [0, 1, 2])
        self.assertIn("recommendations", results[0])
        self.assertIn("errors", results[1])
        self.assertIn("errors", results[2])

    def test_batch_endpoint_validates_the_envelope(self):
        for payload in ({"items": []}, {"items": [VALID] * 101}, {"nope": 1}, [1]):
            self.assertEqual(self.client.post("/api/predict/batch", json=payload).status_code, 400)

    def test_request_ids_are_generated_and_echoed(self):
        self.assertEqual(len(self.client.get("/health").headers["X-Request-ID"]), 12)
        echoed = self.client.get("/health", headers={"X-Request-ID": "abc-123"})
        self.assertEqual(echoed.headers["X-Request-ID"], "abc-123")
        unsafe = self.client.get("/health", headers={"X-Request-ID": "bad id;<x>"})  # not in [A-Za-z0-9._-]
        self.assertNotIn("bad", unsafe.headers["X-Request-ID"])
        self.assertEqual(len(unsafe.headers["X-Request-ID"]), 12)

    def test_rate_limit_returns_429_with_retry_after(self):
        limited = create_app(self.root / "models" / "crop_model.joblib", self.root / "reports", rate_limit=3).test_client()
        codes = [limited.post("/api/predict", json=VALID).status_code for _ in range(4)]
        self.assertEqual(codes, [200, 200, 200, 429])
        blocked = limited.post("/api/predict", json=VALID)
        self.assertEqual(blocked.status_code, 429)
        self.assertGreaterEqual(int(blocked.headers["Retry-After"]), 1)
        self.assertEqual(limited.get("/health").status_code, 200)  # only POSTs are limited

    def test_openapi_spec_matches_the_validator(self):
        spec = self.client.get("/openapi.json").get_json()
        self.assertEqual(spec["openapi"], "3.0.3")
        self.assertEqual(spec, build_spec())  # the endpoint serves exactly what the module builds
        for path in ["/api/predict", "/api/predict/batch", "/api/model", "/health", "/metrics"]:
            self.assertIn(path, spec["paths"])
        props = spec["components"]["schemas"]["CropInput"]["properties"]
        self.assertEqual(sorted(props), sorted(FEATURES))
        self.assertEqual(props["ph"]["maximum"], 14)

    def test_pages_have_no_inline_styles_or_scripts(self):
        """The strict CSP blocks them, so the templates must never emit them."""
        pages = [
            self.client.get("/").get_data(as_text=True),
            self.client.get("/about").get_data(as_text=True),
            self.client.post("/", data={k: str(v) for k, v in VALID.items()}).get_data(as_text=True),
        ]
        for html in pages:
            for forbidden in ('style="', "<style", "onclick=", "onload=", "javascript:"):
                self.assertNotIn(forbidden, html)

    def test_result_page_renders_the_visual_components(self):
        html = self.client.post("/", data={k: str(v) for k, v in VALID.items()}).get_data(as_text=True)
        for marker in ["donut-fg", "radar-user", "radar-typical", "effect-bar", "Plausible crops", "rank-item"]:
            self.assertIn(marker, html)
        self.assertEqual(len(re.findall(r'class="field(?: has-error)?"', html)), 7)  # one card per input

    def test_report_images_are_served(self):
        response = self.client.get("/reports/confusion_matrix.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")


class TestMetricsRegistry(unittest.TestCase):
    def test_histogram_is_cumulative_and_consistent(self):
        m = Metrics()
        for seconds in (0.001, 0.02, 0.3, 10.0):
            m.observe_request("GET", "/", 200, seconds)
        text = m.render(model_loaded=False)
        self.assertIn('le="0.005"} 1', text)
        self.assertIn('le="0.025"} 2', text)
        self.assertIn('le="+Inf"} 4', text)
        self.assertIn('crop_http_request_duration_seconds_count{endpoint="/"} 4', text)
        self.assertIn("crop_model_loaded 0", text)


class TestRateLimiter(unittest.TestCase):
    def test_window_slides_and_keys_are_independent(self):
        now = [0.0]
        limiter = RateLimiter(limit=2, window=10.0, clock=lambda: now[0])
        self.assertEqual(limiter.check("a"), (True, 0))
        self.assertEqual(limiter.check("a"), (True, 0))
        allowed, retry = limiter.check("a")
        self.assertFalse(allowed)
        self.assertEqual(retry, 10)
        self.assertTrue(limiter.check("b")[0])
        now[0] = 10.5
        self.assertTrue(limiter.check("a")[0])

    def test_zero_disables_the_limiter(self):
        limiter = RateLimiter(limit=0)
        self.assertTrue(all(limiter.check("x")[0] for _ in range(1000)))


class TestChartGeometry(unittest.TestCase):
    RANGES = {f: [0.0, 100.0] for f in FEATURES}

    def test_radar_has_one_axis_per_feature_and_scales_to_the_range(self):
        user = {f: 50.0 for f in FEATURES}
        typical = {f: 100.0 for f in FEATURES}
        chart = radar(FEATURES, {f: f for f in FEATURES}, self.RANGES, user, typical)
        self.assertEqual(len(chart["axes"]), len(FEATURES))
        self.assertEqual(len(chart["rings"]), 4)
        self.assertEqual(len(chart["user"].split()), len(FEATURES))
        first_user = [float(v) for v in chart["user"].split()[0].split(",")]
        first_typical = [float(v) for v in chart["typical"].split()[0].split(",")]
        self.assertAlmostEqual(chart["cy"] - first_user[1], 50.0, delta=0.2)      # half the radius, straight up
        self.assertAlmostEqual(chart["cy"] - first_typical[1], 100.0, delta=0.2)  # full radius

    def test_values_outside_the_range_are_clipped(self):
        user = {f: 1000.0 for f in FEATURES}
        chart = radar(FEATURES, {f: f for f in FEATURES}, self.RANGES, user, user)
        self.assertEqual(chart["user"], chart["typical"])
        top = [float(v) for v in chart["user"].split()[0].split(",")]
        self.assertAlmostEqual(chart["cy"] - top[1], 100.0, delta=0.2)

    def test_diverging_bars(self):
        bars = diverging_bars([0.2, -0.1, 0.0])
        self.assertEqual(bars[0], {"x": 50.0, "w": 48.0})
        self.assertEqual(bars[1], {"x": 26.0, "w": 24.0})
        self.assertEqual(bars[2]["w"], 0.0)

    def test_range_marker(self):
        marker = range_marker(10, 60, 0, 100)
        self.assertEqual(marker, {"x": 10.0, "w": 50.0})
        self.assertLessEqual(range_marker(95, 100, 0, 100)["x"] + range_marker(95, 100, 0, 100)["w"], 100.0)


class TestWithoutModel(unittest.TestCase):
    def test_app_degrades_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = create_app(Path(tmp) / "missing.joblib", tmp).test_client()
            self.assertEqual(client.get("/").status_code, 200)
            self.assertIn(b"python train.py", client.get("/").data)
            self.assertEqual(client.post("/api/predict", json=VALID).status_code, 503)
            self.assertEqual(client.get("/api/model").status_code, 503)
            self.assertEqual(client.get("/health").get_json()["model_loaded"], False)


if __name__ == "__main__":
    unittest.main()
