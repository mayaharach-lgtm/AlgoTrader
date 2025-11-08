# AlgoTrader — Machine Learning Stock Prediction System

AlgoTrader is a personal project I built to predict short-term stock movements using machine learning and technical analysis.  
The system learns from real market data and tries to forecast whether a stock will go **up or down the next day**.

---

## Project Overview
The model uses historical stock data from "Yahoo Finance" and extracts a wide range of technical indicators, including:

- Simple & Exponential Moving Averages (SMA, EMA)  
- MACD and RSI (momentum indicators)  
- Bollinger Bands (volatility)  
- Volume changes and ratios  
- Trend slope and rolling statistics  

After feature generation, an XGBoost Classifier is trained, optimized, and evaluated using metrics like Accuracy, Recall, F1, and AUC.  
The final trained model is saved as `xgb_model.pkl` for later use or deployment.


✅ Best parameters found: {'subsample': 0.9, 'reg_lambda': 2.0, 'n_estimators': 800, 'min_child_weight': 5, 'max_depth': 4, 'learning_rate': 0.05, 'gamma': 0.1, 'colsample_bytree': 1.0}
Optimal threshold: 0.101
Accuracy: 58.75%
Precision: 55.56%
Recall: 87.50%
F1-score: 67.96%
AUC: 0.634
Confusion Matrix: TN=12, FP=28, FN=5, TP=35


---

## ⚙️ Installation and Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/AlgoTrader.git
cd AlgoTrader

python -m venv venv
venv\Scripts\activate        # on Windows
# or
source venv/bin/activate     # on macOS / Linux

pip install -r requirements.txt


#Run options:
python main.py              # Simulate with last 6 months of data
python main_live.py         # Live predictions (Buy / Sell / Hold)
streamlit run app.py        # Streamlit web dashboard with live charts

#Run with docker:
docker build -t trading-app .
docker run -p 8501:8501 trading-app

docker save -o trading-app.tar trading-app
docker load -i trading-app.tar

git status
git add .
git commit -m "Update model & README"
git push origin main
