"""Simple ML baseline — linear regression over per-GW features.

Trains on the current season's element-summary history (already cached
by backtest.py). Predicts next-GW points from per-GW features at the time
of that GW.

Why no LightGBM/sklearn dependency:
  - LightGBM would require a much bigger dependency footprint
  - For a single-season's worth of data per player, ridge regression on
    7 features captures most of the signal — what we want is a *baseline*
    to compare against the heuristic, not a state-of-the-art predictor
  - Pure numpy works on Render free without extra installs

Output: predictions per player + a "agreement-with-heuristic" diagnostic.
The point is to show in UI: "heuristic says Salah 7.2, ML baseline says 6.4"
so we know when our heuristic is potentially overconfident.
"""
from __future__ import annotations

from typing import Any

import httpx

from .backtest import _load_cache, _save_cache, fetch_element_summary
from .models import Player

FEATURE_NAMES = [
    "form_proxy",       # rolling pts L3 GW
    "xG",               # cumulative or rolling
    "xA",
    "ict_index",
    "minutes_pct",      # 0-1
    "bps_per_game",
    "is_home",          # 1 if home next, else 0
]


def _ridge_solve(X: list[list[float]], y: list[float], lam: float = 1.0) -> list[float]:
    """Closed-form ridge regression: w = (X^T X + lam I)^-1 X^T y.

    Pure-Python implementation to avoid numpy dependency.
    For small d (number of features), this is fine.
    """
    n = len(X)
    d = len(X[0]) if X else 0
    if n == 0 or d == 0:
        return [0.0] * d

    # Compute XtX (d x d) and Xty (d)
    XtX = [[0.0] * d for _ in range(d)]
    Xty = [0.0] * d
    for i in range(n):
        for a in range(d):
            Xty[a] += X[i][a] * y[i]
            for b in range(d):
                XtX[a][b] += X[i][a] * X[i][b]
    for a in range(d):
        XtX[a][a] += lam

    # Solve XtX w = Xty via Gaussian elimination
    aug = [row + [Xty[i]] for i, row in enumerate(XtX)]
    for col in range(d):
        # Pivot
        pivot = max(range(col, d), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-9:
            continue
        for r in range(d):
            if r == col:
                continue
            factor = aug[r][col] / aug[col][col]
            for c in range(col, d + 1):
                aug[r][c] -= factor * aug[col][c]
    weights = []
    for i in range(d):
        if abs(aug[i][i]) < 1e-9:
            weights.append(0.0)
        else:
            weights.append(aug[i][d] / aug[i][i])
    return weights


def _features_from_history(
    history: list[dict],
    target_idx: int,
    rolling_window: int = 3,
) -> list[float] | None:
    """Build feature row using rows 0..target_idx-1, label = row[target_idx]."""
    if target_idx <= 0 or target_idx >= len(history):
        return None
    prior = history[max(0, target_idx - rolling_window):target_idx]
    if not prior:
        return None
    target = history[target_idx]

    avg_pts = sum(float(h.get("total_points", 0) or 0) for h in prior) / len(prior)
    sum_xG = sum(float(h.get("expected_goals", 0) or 0) for h in prior)
    sum_xA = sum(float(h.get("expected_assists", 0) or 0) for h in prior)
    avg_ict = sum(float(h.get("ict_index", 0) or 0) for h in prior) / len(prior)
    avg_min = sum(float(h.get("minutes", 0) or 0) for h in prior) / len(prior) / 90.0
    avg_bps = sum(float(h.get("bps", 0) or 0) for h in prior) / len(prior)
    is_home = 1.0 if target.get("was_home") else 0.0
    return [avg_pts, sum_xG, sum_xA, avg_ict, avg_min, avg_bps, is_home]


def train_baseline(
    players: list[Player],
    *,
    top_pool: int = 100,
    rolling_window: int = 3,
) -> dict[str, Any]:
    """Train a ridge model on cached element-summary histories.

    Returns:
        {
          "weights": [float, ...],   # one per FEATURE_NAMES
          "intercept": float,
          "n_samples": int,
          "rmse": float,
          "feature_names": [...],
        }
    """
    pool = sorted(players, key=lambda p: p.composite_score, reverse=True)[:top_pool]
    cache = _load_cache()

    X: list[list[float]] = []
    y: list[float] = []

    with httpx.Client(timeout=30) as client:
        for p in pool:
            data = fetch_element_summary(client, p.id, cache)
            if not data:
                continue
            hist = data.get("history", [])
            for i in range(len(hist)):
                feats = _features_from_history(hist, i, rolling_window)
                if feats is None:
                    continue
                X.append([1.0] + feats)  # intercept term
                y.append(float(hist[i].get("total_points", 0) or 0))
    _save_cache(cache)

    if not X:
        return {"error": "No training data"}

    weights = _ridge_solve(X, y, lam=2.0)
    # Compute training RMSE
    sse = 0.0
    for xi, yi in zip(X, y):
        pred = sum(w * x for w, x in zip(weights, xi))
        sse += (pred - yi) ** 2
    rmse = (sse / len(y)) ** 0.5

    return {
        "intercept": round(weights[0], 4),
        "weights": [round(w, 4) for w in weights[1:]],
        "feature_names": FEATURE_NAMES,
        "n_samples": len(y),
        "rmse": round(rmse, 3),
    }


def predict(player: Player, model: dict[str, Any]) -> float | None:
    """Predict next-GW points for a player using a trained baseline model.

    Pulls latest history rows from cache; returns None if not enough history.
    """
    if "weights" not in model:
        return None
    cache = _load_cache()
    cached = cache.get(str(player.id))
    if not cached:
        return None
    hist = (cached.get("data") or {}).get("history", [])
    if len(hist) < 3:
        return None
    feats = _features_from_history(hist, len(hist) - 1, rolling_window=3)
    if feats is None:
        return None
    full = [1.0] + feats
    pred = model["intercept"] + sum(w * x for w, x in zip(model["weights"], full[1:]))
    return round(max(0, pred), 2)


def compare_to_heuristic(
    players: list[Player],
    model: dict[str, Any],
    *,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """For top-N players by composite score, show heuristic rank vs ML rank.

    Useful as a sanity check: do they generally agree? Where do they disagree?
    """
    pool = sorted(players, key=lambda p: p.composite_score, reverse=True)[:top_n]
    out = []
    for p in pool:
        ml_pred = predict(p, model)
        out.append({
            "id": p.id,
            "name": p.name,
            "photo": p.photo,
            "position": p.position_name,
            "composite_score": round(p.composite_score, 3),
            "ep_next_fpl": p.ep_next,
            "ml_baseline_pred": ml_pred,
            "agreement_delta": (
                round(ml_pred - p.ep_next, 2) if ml_pred is not None and p.ep_next else None
            ),
        })
    return out
