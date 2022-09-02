import streamlit as st
import numpy as np
import pandas as pd
import datetime as datetime
import pandas_datareader
import datetime
import plotly.graph_objects as go
import sklearn
import sklearn.linear_model
import sklearn.model_selection
from PIL import Image
import yfinance as yf

st.title('AIで株価予測アプリ')
st.write('AIを使って、株価を予測してみましょう。')

image = Image.open('stock_predict.png')
st.image(image, use_column_width=True)

st.write('※あくまでAIによる予測です（参考値）。こちらのアプリによる損害や損失は一切保証しかねます。')

st.header('株価銘柄のティッカーシンボルを入力してください。')
stock_name = st.text_input('例：AAPL, FB, SFTBY（大文字、小文字どちらでも可）', 'AAPL')

stock_name = stock_name.upper()

stock_name_full = yf.Ticker(str(stock_name))
stock_name_full = stock_name_full.info['longName']

link = 'https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_usequity_list.html'
st.markdown(link)
st.write('ティッカーシンボルについては上のリンク（SBI証券）をご参照ください。')
try:
    df_stock = pandas_datareader.data.DataReader(stock_name, 'yahoo', '2021-01-01')

    st.header(stock_name_full + ' 2021年1月1日から現在までの価格（USD）')
    st.write(df_stock)

    st.header(stock_name_full + ' 終値と14日間平均（USD）')
    df_stock['SMA'] = df_stock['Close'].rolling(window=14).mean()
    df_stock2 = df_stock[['Close', 'SMA']]
    st.line_chart(df_stock2)

    st.header(stock_name_full + ' 値動き（USD）')
    df_stock['change'] = (((df_stock['Close'] - df_stock['Open'])) / (df_stock['Open']) * 100)
    st.line_chart(df_stock['change'].tail(100))

    fig = go.Figure(
        data = [go.Candlestick(
            x = df_stock.index,
            open = df_stock['Open'],
            high = df_stock['High'],
            low = df_stock['Low'],
            close = df_stock['Close'],
            increasing_line_color = 'green',
            decreasing_line_color = 'red',
            )
        ]
    )

    st.header(stock_name_full + ' キャンドルスティック')
    st.plotly_chart(fig)

    df_stock['label'] = df_stock['Close'].shift(-30)

    st.header(stock_name_full + ' 1か月後を予測しよう(USD)')
    def stock_predict():
        # 機械学習（ディープラーニング）
        X = np.array(df_stock.drop(['label', 'SMA'], axis=1))
        X = sklearn.preprocessing.scale(X)
        predict_data = X[-30:]
        X = X[:-30]
        y = np.array(df_stock['label'])
        y = y[:-30]
        # データの分割
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        # 訓練データを用いて学習する
        model = sklearn.linear_model.LinearRegression()
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        # 小数点第一位で四捨五入
        st.write(f'正答率は{round((accuracy) * 100, 1)}%です。')

        # accuracyより信頼度を表示
        if accuracy > 0.75:
            st.write('信頼度：高')
        elif accuracy > 0.5:
            st.write('信頼度：中')
        else:
            st.write('信頼度：低')
        st.write('オレンジの線（predict）が予測値です。')

        # 検証データを用いて検証してみる
        predict_data = model.predict(predict_data)
        df_stock['Predict'] = np.nan
        last_date = df_stock.iloc[-1].name
        one_day = 86400
        next_unix = last_date.timestamp() + one_day

        for data in predict_data:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_day
            df_stock.loc[next_date] = np.append([np.nan]* (len(df_stock.columns)-1), data)

        df_stock['Close'].plot(figsize=(15, 6), color='green')
        df_stock['Predict'].plot(figsize=(15, 6), color='orange')

        df_stock3 = df_stock[['Close', 'Predict']]
        st.line_chart(df_stock3)

    # ボタンを押すとstock_predict()が発動
    if st.button('予測する'):
        stock_predict()

except:
    st.error(
        'エラーが起きているようです。'
    )
st.write('Copyright © 2021 Tomoyuki Yoshikawa. All Rights Reserved.')
