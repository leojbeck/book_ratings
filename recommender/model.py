# Trains a Ridge regression to predict rating differential (my_rating - gr_avg).
# Predicted rating = gr_avg + predicted_differential, clamped to [1, 5].
#
# Usage (from project root):
#   py recommender/model.py          <- train, evaluate, save recommender/model.json
#   py recommender/model.py --eval   <- load saved model and print evaluation only

import json
import os
import sys
from datetime import date

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

# sibling import — works whether run as script or imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (FEATURE_NAMES, build_features, build_stats, build_tfidf,
                      cache_key, load_cache, load_read_books)

_HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(_HERE, "model.json")

# ---------- data prep ----------

def build_xy(read_books, stats, cache, tfidf_model):
    """
    Build feature matrix X and differential target y.
    Skips books missing either rating — should be none in the read CSV but safe to guard.
    """
    X, y, used = [], [], []
    for b in read_books:
        if b["My Rating"] is None or b["Avg Rating"] is None:
            continue
        feats = build_features(b, stats, cache, tfidf_model)
        X.append([feats[f] for f in FEATURE_NAMES])
        y.append(b["My Rating"] - b["Avg Rating"])
        used.append(b)
    return np.array(X, dtype=float), np.array(y, dtype=float), used

# ---------- train ----------

def train(read_books=None, cache=None):
    """
    Fit model, run 5-fold CV for error estimates, save params to model.json.
    Returns (params_dict, stats, tfidf_model) so recommend.py can reuse them.
    """
    if read_books is None:
        read_books = load_read_books()
    if cache is None:
        cache = load_cache()

    stats = build_stats(read_books)
    tfidf = build_tfidf(read_books, cache)
    X, y, used = build_xy(read_books, stats, cache, tfidf)

    scaler   = StandardScaler()
    X_sc     = scaler.fit_transform(X)

    # RidgeCV with LOO (cv=None) for alpha selection — log-spaced grid
    alphas = np.logspace(-3, 3, 60)
    ridge  = RidgeCV(alphas=alphas, cv=None)
    ridge.fit(X_sc, y)

    # 5-fold CV for honest error reporting (not used for fitting)
    kf   = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv = cross_val_predict(RidgeCV(alphas=alphas), X_sc, y, cv=kf)

    # Per-book residuals for diagnostics
    y_hat   = ridge.predict(X_sc)
    resids  = (y - y_hat).tolist()

    params = {
        "feature_names": FEATURE_NAMES,
        "scaler_mean":   scaler.mean_.tolist(),
        "scaler_std":    scaler.scale_.tolist(),
        "coefficients":  ridge.coef_.tolist(),
        "intercept":     float(ridge.intercept_),
        "best_alpha":    float(ridge.alpha_),
        "train_mae":     float(mean_absolute_error(y, y_hat)),
        "train_rmse":    float(np.sqrt(np.mean((y - y_hat) ** 2))),
        "cv_mae":        float(mean_absolute_error(y, y_cv)),
        "cv_rmse":       float(np.sqrt(np.mean((y - y_cv) ** 2))),
        # residuals keyed by title for later inspection
        "residuals":     {used[i]["Title"]: round(resids[i], 3) for i in range(len(used))},
        # MAE if we always predict diff=0 (i.e. just trust GR avg, no personalisation)
        "baseline_mae":  float(np.mean(np.abs(y))),
        "n_books":       len(used),
        "trained_at":    str(date.today()),
    }

    with open(MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    return params, stats, tfidf

# ---------- inference ----------

def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("model.json not found — run `py recommender/model.py` first")
    with open(MODEL_FILE, encoding="utf-8") as f:
        return json.load(f)

def predict(book, stats, cache, tfidf_model, params):
    """
    Predict (differential, predicted_rating) for one book.
    predicted_rating = gr_avg + differential, clamped to [1, 5].
    """
    feats = build_features(book, stats, cache, tfidf_model)
    x     = np.array([feats[f] for f in params["feature_names"]], dtype=float)
    x_sc  = (x - np.array(params["scaler_mean"])) / np.array(params["scaler_std"])
    diff  = float(np.dot(x_sc, params["coefficients"]) + params["intercept"])
    gr    = feats["gr_avg"]
    rating = max(1.0, min(5.0, gr + diff))
    return diff, rating

# ---------- diagnostics ----------

def _print_results(params):
    print(f"\nTrained on {params['n_books']} books   best alpha={params['best_alpha']:.4f}")
    print(f"\n  Train  MAE={params['train_mae']:.3f}  RMSE={params['train_rmse']:.3f}")
    print(f"  5-fold MAE={params['cv_mae']:.3f}  RMSE={params['cv_rmse']:.3f}")
    print(f"  (naive baseline MAE predicting diff=0: {params['baseline_mae']:.3f})")

    coefs = list(zip(params["feature_names"], params["coefficients"]))
    print(f"\nFeature coefficients (sorted by |weight|, standardised units):")
    for name, coef in sorted(coefs, key=lambda x: abs(x[1]), reverse=True):
        bar = ("+" if coef >= 0 else "-") * min(20, int(abs(coef) * 8))
        print(f"  {name:<20} {coef:+.4f}  {bar}")
    print(f"  {'intercept':<20} {params['intercept']:+.4f}")

    # Largest residuals — books the model got most wrong
    resids = params["residuals"]
    worst  = sorted(resids.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"\nLargest training residuals (actual - predicted differential):")
    for title, r in worst:
        print(f"  {r:+.3f}  {title}")

# ---------- CLI ----------

if __name__ == "__main__":
    if "--eval" in sys.argv:
        _print_results(load_model())
    else:
        print("Training...")
        params, _, _ = train()
        _print_results(params)
        print(f"\nSaved to recommender/model.json")
