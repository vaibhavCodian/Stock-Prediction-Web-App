from flask import Flask, render_template, request, url_for

#for model
import math
import matplotlib.pyplot as plt
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
    close_px.plot(label='AAPL')
    mavg.plot(label='mavg')
    plt.legend()
    # Saving mavg
    figfile = BytesIO()
    plt.savefig(figfile, format='png', transparent=True)
    figfile.seek(0)  # rewind to beginning of file
    figdata_png_m = figfile.getvalue()  # extract string (stream of bytes)
    figdata_png_m = base64.b64encode(figdata_png_m)
    # Prediction Part
    plt.clf()
    # style.use('fast')
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
    # X_lately = X[-forecast_out:]
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    # Linear regression
    clfreg = LinearRegression(n_jobs=-1)
    clfreg.fit(X, y)

    # Quadratic Regression 2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2.fit(X, y)

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

    dfreg['Adj Close'].plot()
    dfreg['Forecast'].plot()
    plt.legend(loc=0)
    plt.xlabel('Date')
    plt.ylabel('Price')

    # Saving Prediction
    figfile = BytesIO()
    plt.savefig(figfile, format='png', transparent=True)
    figfile.seek(0)  # rewind to beginning of file
    figdata_png_p = figfile.getvalue()  # extract string (stream of bytes)
    figdata_png_p = base64.b64encode(figdata_png_p)
    
    return figdata_png_m, figdata_png_p

app = Flask(__name__)

@app.route('/')
def home():
    ticker = request.args.get('ticker')
    if ticker:
        data_m, data_p = stock_l(ticker)
        # return data_p
        return render_template('index.html', data_m=data_m, ticker=ticker, data_p=data_p)
    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run()
