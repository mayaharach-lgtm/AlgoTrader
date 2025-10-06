Dependencies:
streamlit
yfinance
pandas
numpy
plotly
colorama

Git control:
git status                      # check changes
git add .                       # stage all files
git commit -m "comment"
git push origin main            # upload changes to GitHub



Run Localy(wihout Docker) options:
git clone https://github.com/YOUR_USERNAME/AlgoTrader.git
cd AlgoTrader
python -m venv venv
venv\Scripts\activate        # on Windows
# or
source venv/bin/activate     # on macOS / Linux
pip install -r requirements.txt


1. py main.py - while market is closed, showing real data from the last 6 months
or 2. py main_live.py - showing live prices and buy\sell\hold recomendations 
or 3. streamlit run app.py - showing live prices and buy\sell\hold recomendations + graph on website

Running with Docker:
1.docker build -t trading-app .
2.Dokcer: docker run -p 8501:8501 trading-app    

Changes with Docker:
docker save -o trading-app.tar trading-app    #save image for sharing or backup
docker load -i trading-app.tar     #load image on another machine
docker build -t trading-app .     #update after changes 
docker save -o trading-app.tar trading-app      #save update after changes  