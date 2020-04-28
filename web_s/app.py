from flask import Flask, render_template, request, url_for

#for model
import math
import matplotlib.pyplot as plt, mpld3
import pandas as pd
import numpy as np

from sklearn import preprocessing

from pandas_datareader import data

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#__._._._._._._>
import datetime
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from matplotlib import style
# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__
# Adjusting the style of matplotlib
plt.style.use(['fast', 'seaborn-muted'])
plt.tight_layout()


def stock_l(ticker):
    plt.clf()
    #--> Getting Data
    # We would like all available data from 10 month back (approx ~) 
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=(30 * 10))
    # User pandas_reader.data.DataReader to load the desired data. As simple as that.
    df = data.DataReader(ticker, 'yahoo', start_date, end_date)
    # Preprocessing Days and Setting labels
    close = df['Close']
    close_px = df['Adj Close']
    mavg = close_px.rolling(window=100).mean()
    all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
    close = close.reindex(all_weekdays)
    close = close.fillna(method='ffill')
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(close_px.index, close_px, label=ticker)
    ax.plot(mavg.index, mavg, label="m_avg")
    ax.legend()
    fig_mavg = mpld3.fig_to_html(fig)
    
    #  Prediction_____PART

    dfreg = df.loc[:,['Adj Close','Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    dfreg.head()
    # Drop missing value
    dfreg.fillna(value=-9999999, inplace=True)

    # We want to separate 1 percent of the data to forecast
    # forecast_out = int(math.ceil(0.01 * len(dfreg)))
    forecast_out = 60
    #  
    # Separating the label here, we want to predict the AdjClose
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))

    # Scale the X so that everyone can have the same distribution for linear regression
    X = preprocessing.scale(X)

    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    # Linear regression
    clfreg = LinearRegression(n_jobs=-1)
    clfreg.fit(X, y)

    # Making the forecast
    forecast_set = clfreg.predict(X_lately)
    dfreg['Forecast'] = np.nan

    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)

    for i in forecast_set:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]



    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(dfreg['Adj Close'].index, dfreg['Adj Close'], label="Adj Close")
    ax.plot(dfreg['Forecast'].index, dfreg['Forecast'], label="Forecast")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    fig_forecast = mpld3.fig_to_html(fig)
    # print(fig_forecast)

    return fig_mavg, fig_forecast


app = Flask(__name__)

@app.route('/')
def home():
    ticker = request.args.get('ticker')
        
    if ticker:
        ticker = ticker.strip()
        html_m, html_f = stock_l(ticker)
        # print(html_f)
        return render_template('index.html', html_m=html_m, html_f=html_f)
    else:
        return render_template('index.html', ticker=ticker)



if __name__ == '__main__':
    app.run()
