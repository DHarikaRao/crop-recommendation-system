"""Geometry for the server-rendered SVG charts.

Charts are drawn with SVG presentation attributes computed here, so the page needs no chart library
and no inline styles, and the strict Content-Security-Policy stays intact.
"""
import math

RADAR_SIZE = (360, 320)
RADAR_CENTER = (180, 160)
RADAR_RADIUS = 100


def _norm(value: float, low: float, high: float) -> float:
    return 0.0 if high <= low else min(1.0, max(0.0, (value - low) / (high - low)))


def _point(index: int, count: int, radius: float) -> tuple[float, float]:
    angle = -math.pi / 2 + 2 * math.pi * index / count  # start at 12 o'clock, go clockwise
    return RADAR_CENTER[0] + radius * math.cos(angle), RADAR_CENTER[1] + radius * math.sin(angle)


def _points(coords) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)


def radar(features, labels, ranges, user, typical) -> dict:
    """Radar chart comparing the user's inputs with the typical crop profile (both scaled to the training range)."""
    count = len(features)
    axes = []
    for i, feature in enumerate(features):
        x2, y2 = _point(i, count, RADAR_RADIUS)
        lx, ly = _point(i, count, RADAR_RADIUS + 22)
        cos = math.cos(-math.pi / 2 + 2 * math.pi * i / count)
        axes.append({
            "x2": round(x2, 1), "y2": round(y2, 1),
            "lx": round(lx, 1), "ly": round(ly + 4, 1),
            "anchor": "start" if cos > 0.3 else "end" if cos < -0.3 else "middle",
            "label": labels[feature],
        })

    def shape(values):
        return [
            _point(i, count, RADAR_RADIUS * _norm(values[f], *ranges[f]))
            for i, f in enumerate(features)
        ]

    user_shape = shape(user)
    return {
        "width": RADAR_SIZE[0], "height": RADAR_SIZE[1],
        "cx": RADAR_CENTER[0], "cy": RADAR_CENTER[1],
        "rings": [_points([_point(i, count, RADAR_RADIUS * level) for i in range(count)]) for level in (0.25, 0.5, 0.75, 1.0)],
        "axes": axes,
        "user": _points(user_shape),
        "typical": _points(shape(typical)),
        "user_dots": [(round(x, 1), round(y, 1)) for x, y in user_shape],
    }


def diverging_bars(effects) -> list[dict]:
    """x/width (in a 0..100 box, zero line at 50) for signed effects; the largest effect fills 48 units."""
    scale = max([abs(e) for e in effects] + [0.01])
    bars = []
    for effect in effects:
        width = abs(effect) / scale * 48
        bars.append({"x": round(50 if effect >= 0 else 50 - width, 2), "w": round(width, 2)})
    return bars


def range_marker(low: float, high: float, hard_min: float, hard_max: float) -> dict:
    """Where the training range sits inside the slider's allowed range, as x/width in a 0..100 box."""
    span = hard_max - hard_min
    x = _norm(low, hard_min, hard_max) * 100
    width = max(1.5, (_norm(high, hard_min, hard_max) - _norm(low, hard_min, hard_max)) * 100) if span else 100
    return {"x": round(x, 2), "w": round(min(width, 100 - x), 2)}
