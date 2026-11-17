"""A tiny in-process metrics registry that renders the Prometheus text format.

Counters are kept per process. Run a single gunicorn worker (with threads), as the
Dockerfile does, or switch to prometheus_client's multiprocess mode if you need several workers.
"""
import threading
from collections import defaultdict

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests = defaultdict(int)          # (method, endpoint, status) -> count
        self._buckets = {}                         # endpoint -> observations per bucket (+Inf last)
        self._latency_sum = defaultdict(float)
        self._predictions = defaultdict(int)       # crop -> count

    def observe_request(self, method, endpoint, status, seconds):
        with self._lock:
            self._requests[(method, endpoint, str(status))] += 1
            counts = self._buckets.setdefault(endpoint, [0] * (len(LATENCY_BUCKETS) + 1))
            for position, limit in enumerate(LATENCY_BUCKETS):
                if seconds <= limit:
                    counts[position] += 1
                    break
            else:
                counts[-1] += 1
            self._latency_sum[endpoint] += seconds

    def count_prediction(self, crop):
        with self._lock:
            self._predictions[crop] += 1

    def render(self, model_loaded: bool) -> str:
        with self._lock:
            lines = [
                "# HELP crop_http_requests_total Total HTTP requests.",
                "# TYPE crop_http_requests_total counter",
            ]
            for (method, endpoint, status), value in sorted(self._requests.items()):
                lines.append(
                    f'crop_http_requests_total{{method="{method}",endpoint="{endpoint}",status="{status}"}} {value}'
                )
            lines += [
                "# HELP crop_http_request_duration_seconds Request latency in seconds.",
                "# TYPE crop_http_request_duration_seconds histogram",
            ]
            for endpoint, counts in sorted(self._buckets.items()):
                running = 0
                for limit, count in zip(LATENCY_BUCKETS, counts):
                    running += count
                    lines.append(
                        f'crop_http_request_duration_seconds_bucket{{endpoint="{endpoint}",le="{limit}"}} {running}'
                    )
                running += counts[-1]
                lines.append(f'crop_http_request_duration_seconds_bucket{{endpoint="{endpoint}",le="+Inf"}} {running}')
                lines.append(f'crop_http_request_duration_seconds_sum{{endpoint="{endpoint}"}} {self._latency_sum[endpoint]:.6f}')
                lines.append(f'crop_http_request_duration_seconds_count{{endpoint="{endpoint}"}} {running}')
            lines += [
                "# HELP crop_predictions_total Recommendations served, by top crop.",
                "# TYPE crop_predictions_total counter",
            ]
            for crop, value in sorted(self._predictions.items()):
                lines.append(f'crop_predictions_total{{crop="{crop}"}} {value}')
            lines += [
                "# HELP crop_model_loaded 1 if a trained model is loaded.",
                "# TYPE crop_model_loaded gauge",
                f"crop_model_loaded {1 if model_loaded else 0}",
            ]
        return "\n".join(lines) + "\n"
