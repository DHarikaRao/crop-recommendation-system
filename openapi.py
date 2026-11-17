"""OpenAPI 3 description of the JSON API, built from the same limits the validator uses."""
from crop_model import FEATURE_INFO, FEATURES


def build_spec() -> dict:
    inputs = {
        name: {
            "type": "number",
            "minimum": info["min"],
            "maximum": info["max"],
            "description": f"{info['label']} ({info['unit']})",
        }
        for name, info in FEATURE_INFO.items()
    }
    crop = {
        "type": "object",
        "properties": {
            "crop": {"type": "string"},
            "name": {"type": "string"},
            "probability": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    explanation = {
        "type": "object",
        "properties": {
            "feature": {"type": "string"},
            "value": {"type": "number"},
            "typical": {"type": "number"},
            "effect_pp": {"type": "number", "description": "Change in the top crop's probability, in percentage points."},
            "direction": {"type": "string", "enum": ["supports", "lowers", "neutral"]},
        },
    }
    result = {
        "type": "object",
        "properties": {
            "recommendations": {"type": "array", "items": {"$ref": "#/components/schemas/Crop"}},
            "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "plausible_crops": {"type": "array", "items": {"$ref": "#/components/schemas/Crop"}},
            "coverage_target": {"type": "number"},
            "unusual_input": {"type": "boolean"},
            "uncertain": {"type": "boolean", "description": "True when no crop reaches the confidence bar, so the three best guesses are returned."},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "explanation": {"type": "array", "items": {"$ref": "#/components/schemas/Explanation"}},
        },
    }
    error_responses = {
        "400": {"description": "Invalid input; one message per field.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
        "429": {"description": "Rate limit exceeded; see the Retry-After header.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
        "503": {"description": "No trained model is loaded.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
    }
    json_body = lambda ref: {"required": True, "content": {"application/json": {"schema": {"$ref": ref}}}}  # noqa: E731
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Crop Recommendation API",
            "version": "1.0.0",
            "description": "Recommends crops from soil and climate inputs, with confidence sets and explanations.",
        },
        "paths": {
            "/api/predict": {"post": {
                "summary": "Recommend crops for one set of conditions",
                "requestBody": json_body("#/components/schemas/CropInput"),
                "responses": {"200": {"description": "Recommendations", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/PredictResponse"}}}}, **error_responses},
            }},
            "/api/predict/batch": {"post": {
                "summary": "Recommend crops for up to 100 sets of conditions",
                "requestBody": json_body("#/components/schemas/BatchRequest"),
                "responses": {"200": {"description": "One result (or field errors) per item", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/BatchResponse"}}}}, **error_responses},
            }},
            "/api/model": {"get": {"summary": "Selected model and held-out metrics", "responses": {"200": {"description": "Model summary"}, "503": error_responses["503"]}}},
            "/health": {"get": {"summary": "Liveness and model status", "responses": {"200": {"description": "OK"}}}},
            "/metrics": {"get": {"summary": "Prometheus metrics", "responses": {"200": {"description": "Prometheus text format"}}}},
        },
        "components": {"schemas": {
            "CropInput": {"type": "object", "required": FEATURES, "properties": inputs},
            "Crop": crop,
            "Explanation": explanation,
            "PredictResponse": result,
            "BatchRequest": {"type": "object", "required": ["items"], "properties": {
                "items": {"type": "array", "minItems": 1, "maxItems": 100, "items": {"$ref": "#/components/schemas/CropInput"}}}},
            "BatchResponse": {"type": "object", "properties": {"results": {"type": "array", "items": {"type": "object"}}}},
            "Error": {"type": "object", "properties": {"error": {"type": "string"}, "errors": {"type": "object", "additionalProperties": {"type": "string"}}}},
        }},
    }
