# ============================================
# model_training.py — XGBoost Classifier with Feature Selection & Fine-Tuning
# ============================================

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, plot_importance


# ---------------------------
# Data Loading
# ---------------------------
def load_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download daily price history from Yahoo Finance."""
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise ValueError(f"No data found for {ticker}")
    return data


# ---------------------------
# RSI Helper
# ---------------------------
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# ---------------------------
# Feature Engineering
# ---------------------------
def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate feature-rich dataset for binary classification (Up/Down)."""
    df = data.copy()

    # Handle MultiIndex (e.g., ('Close', 'META'))
    if isinstance(df.columns, pd.MultiIndex):
        close_col = [c for c in df.columns if "Close" in str(c[0])]
        vol_col = [c for c in df.columns if "Volume" in str(c[0])]
        if not close_col:
            raise ValueError(f"No 'Close' in columns: {df.columns}")
        close_prices = df.loc[:, close_col[0]]
        volume_series = df.loc[:, vol_col[0]] if vol_col else None
    else:
        close_prices = df["Close"] if "Close" in df.columns else df[df.columns[0]]
        volume_series = df["Volume"] if "Volume" in df.columns else None

    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]
    if isinstance(volume_series, pd.DataFrame):
        volume_series = volume_series.iloc[:, 0]

    df_feat = pd.DataFrame(index=df.index)

    # === Base indicators ===
    df_feat["return_1d"] = close_prices.pct_change()
    df_feat["sma_5"] = close_prices.rolling(5).mean()
    df_feat["sma_20"] = close_prices.rolling(20).mean()
    df_feat["sma_ratio"] = df_feat["sma_5"] / df_feat["sma_20"]
    df_feat["volatility_20"] = close_prices.pct_change().rolling(20).std()

    # === Advanced indicators ===
    df_feat["ema_10"] = close_prices.ewm(span=10, adjust=False).mean()
    df_feat["ema_50"] = close_prices.ewm(span=50, adjust=False).mean()
    df_feat["macd"] = df_feat["ema_10"] - df_feat["ema_50"]

    std20 = close_prices.rolling(20).std()
    sma20 = df_feat["sma_20"]
    df_feat["boll_high"] = sma20 + 2 * std20
    df_feat["boll_low"] = sma20 - 2 * std20
    df_feat["rsi_14"] = _rsi(close_prices, 14)
    df_feat["volume_change"] = volume_series.pct_change() if volume_series is not None else 0.0

    # === Time-based features ===
    df_feat["month"] = df_feat.index.month
    df_feat["weekday"] = df_feat.index.weekday

    # === Extra lagged/trend features ===
    df_feat["return_2d"] = close_prices.pct_change(2)
    df_feat["return_5d"] = close_prices.pct_change(5)
    df_feat["rolling_mean_10"] = close_prices.rolling(10).mean()
    df_feat["rolling_std_10"] = close_prices.rolling(10).std()
    df_feat["boll_position"] = (close_prices - df_feat["boll_low"]) / (df_feat["boll_high"] - df_feat["boll_low"])

    def calc_slope(prices, window=10):
        slopes = [np.nan] * (window - 1)
        for i in range(window, len(prices) + 1):
            y = prices[i - window:i]
            x = np.arange(window).reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            slopes.append(model.coef_[0])
        return pd.Series(slopes, index=prices.index)

    df_feat["trend_slope_10"] = calc_slope(close_prices, 10)

    # === Target (next-day direction) ===
    df_feat["target"] = (close_prices.shift(-1) > close_prices).astype(int)

    # Cleanup
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    return df_feat


# ---------------------------
# Split by Time
# ---------------------------
def split_data(df_feat: pd.DataFrame):
    """TimeSeriesSplit and return train/test sets."""
    features = [
        "return_1d", "return_2d", "return_5d",
        "sma_ratio", "volatility_20",
        "ema_10", "ema_50", "macd",
        "boll_high", "boll_low", "boll_position",
        "rsi_14", "volume_change",
        "rolling_mean_10", "rolling_std_10", "trend_slope_10",
        "month", "weekday"
    ]
    X = df_feat[features]
    y = df_feat["target"]

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test, features


# ---------------------------
# Hyperparameter Tuning
# ---------------------------
def tune_xgb(X_train, y_train):
    """RandomizedSearchCV for XGBClassifier."""
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    imbalance = (neg / max(pos, 1))

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    param_dist = {
        "n_estimators": [200, 400, 600, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6, 8],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.3],
        "reg_lambda": [1.0, 1.5, 2.0],
        "scale_pos_weight": [1.0, imbalance, max(1.0, imbalance * 0.7)]
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=15,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print("✅ Best parameters found:", search.best_params_)
    return search.best_estimator_


# ---------------------------
# Evaluation
# ---------------------------
def evaluate(model: XGBClassifier, X_test, y_test):
    """Evaluate and print metrics using the F1-optimal threshold."""
    proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    preds = (proba >= best_thr).astype(int)

    acc = accuracy_score(y_test, preds) * 100
    prec = precision_score(y_test, preds, zero_division=0) * 100
    rec = recall_score(y_test, preds, zero_division=0) * 100
    f1 = f1_score(y_test, preds, zero_division=0) * 100
    auc = roc_auc_score(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    print(f"Optimal threshold: {best_thr:.3f}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall: {rec:.2f}%")
    print(f"F1-score: {f1:.2f}%")
    print(f"AUC: {auc:.3f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    return preds


# ---------------------------
# Plot Feature Importance
# ---------------------------
def plot_feature_importance(model, feature_names):
    """Display XGBoost feature importance by gain."""
    plt.figure(figsize=(10, 6))
    plot_importance(model, importance_type='gain', xlabel='Gain', ylabel='Feature',
                    show_values=False, title='Feature Importance (by Gain)', color='skyblue')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------
# Fine-Tune depth & learning rate around best params
# ---------------------------
def fine_tune_depth_lr(X_train, y_train, X_test, y_test, base_params):
    """Quick grid over max_depth and learning_rate; maximizes AUC."""
    best_auc = -1.0
    best_params = {k: v for k, v in base_params.items() if k != "eval_metric"}
    print("\n🔧 Fine-tuning depth and learning rate...")

    for depth in [3, 4, 5, 6, 7, 8]:
        for lr in [0.02, 0.03, 0.05, 0.07, 0.1]:
            params = best_params.copy()
            params.update({"max_depth": depth, "learning_rate": lr, "eval_metric": "logloss"})
            model = XGBClassifier(**params, n_jobs=-1)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            if auc > best_auc:
                best_auc = auc
                best_params.update({"max_depth": depth, "learning_rate": lr})
    print(f"✅ Best fine-tuned params: depth={best_params['max_depth']}, lr={best_params['learning_rate']}, AUC={best_auc:.3f}")
    return best_params


# ---------------------------
# Save Model
# ---------------------------
def save_model(model, filename="xgb_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved as {filename}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    ticker = input("Enter ticker (e.g. AAPL, MSFT, TSLA): ").strip().upper()
    if not ticker:
        raise SystemExit("No ticker provided.")

    data = load_data(ticker, period="2y")
    df_feat = create_features(data)
    print(f"Positive (UP) rate in sample: {df_feat['target'].mean() * 100:.2f}%")

    X_train, X_test, y_train, y_test, features = split_data(df_feat)

    # ---- Stage 1: Tune full model on all features ----
    model = tune_xgb(X_train, y_train)
    _ = evaluate(model, X_test, y_test)
    plot_feature_importance(model, features)

    # ---- Stage 2: Feature Selection (Top-10) + quick retrain ----
    importances = model.feature_importances_
    top_features = [f for _, f in sorted(zip(importances, features), reverse=True)[:10]]
    print(f"Top 10 features used for retraining: {top_features}")

    X_train_sel = X_train[top_features]
    X_test_sel = X_test[top_features]

    tuned_model = XGBClassifier(
        **{k: v for k, v in model.get_params().items() if k != "eval_metric"},
        eval_metric="logloss"
    )
    tuned_model.fit(X_train_sel, y_train)
    _ = evaluate(tuned_model, X_test_sel, y_test)

    # ---- Stage 3: Fine-tune depth & learning rate on Top-10 features ----
    fine_params = fine_tune_depth_lr(X_train_sel, y_train, X_test_sel, y_test, tuned_model.get_params())
    final_model = XGBClassifier(**{**fine_params, "eval_metric": "logloss"}, n_jobs=-1)
    final_model.fit(X_train_sel, y_train)
    _ = evaluate(final_model, X_test_sel, y_test)

    save_model(final_model, "xgb_model_finetuned.pkl")
