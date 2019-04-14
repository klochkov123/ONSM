import numpy as np
from pandas import read_csv, read_json, read_excel, DataFrame, to_datetime
import csv
from math import sqrt


def find_homo_Be(x, scale=4, res=20):
    assert(np.size(x) > 2 * res)

    a = (np.abs(x) > 0.000001).astype(dtype=np.float32)

    def check_interval(l, r):
        p = np.mean(a[l:r])
        cr_val = sqrt(p * (1-p) / res) * scale
        sums = np.abs(
            [np.mean(a[t:t + res]) - p for t in range(l, r)]
        )

        return np.all(sums <= cr_val)

    left = 0
    right = np.size(x) - res

    while right - left > res:
        split = left + (right - left) // 2
        if check_interval(split, right):
            right = split
        else:
            left = split

    return left


def read_SIFI(name_to_sifi):
    df = read_excel(name_to_sifi)
    X = []
    names = []
    for name in df.columns.values.tolist():
        if name not in ['Date', 'Year', 'Month', 'Day']:
            names.append(name)
            col = df[name]
            X.append(col)
    X = np.array(X, dtype=float)[:, :-1]
    X = X[:, 750:]

    return X, names


def read_stock_twits_user_sentiment(name_to_twits, min_delta=0.1, min_days=200):
    df = read_csv(name_to_twits, index_col='user_id')
    names = df.index

    #avoid_empty = lambda x, err_t: x[0] if np.size(x) > 0 else err_t
    starts = [find_homo_Be(df.values[i, :]) for i, _ in enumerate(names)]
    deltas = [0.0 if start == -1
              else np.mean((np.abs(df.values[i, start:]) > 0.00001).astype(dtype=np.float32))
              for i, start in enumerate(starts)]
    idx = [i for i, (delta, start) in enumerate(zip(deltas, starts))
           if delta >= min_delta and start <= np.shape(df.values)[1] - min_days]

    return df.values[idx, np.max(np.array(starts)[idx]):], np.array(deltas)[idx], names[idx]


def read_crix_returns(name_to_crix_price):
    crix_price = read_json('crix.json', convert_dates=True)
    crix_price = crix_price.loc[crix_price['price'] != 'NA']

    crix_price.set_index('date', inplace=True)

    vals = np.reshape(crix_price.values, (np.size(crix_price.values),))

    vals = np.array(vals, dtype='float')
    returns = np.log(np.divide(vals[1:], vals[:-1]))

    crix_returns = DataFrame(
        data=np.reshape(returns, (np.size(returns), 1)),
        columns=['return'],
        index=crix_price.index[1:]
    )

    return crix_returns


def read_crypto_sentiment(name_to_crypto_sentiment, top_num=50):
    with open('currency_sentiment_full.csv') as csv_file:
        csv_reader = csv.reader(csv_file)

        i = 0
        headers = []
        for row in csv_reader:
            i += 1
            if i > 3:
                break
            headers.append(np.array(row))

        sent_idx = np.where(headers[1] == 'sentiment_GL')
        columns = headers[0][sent_idx]

        messages_idx = np.where(headers[1] == 'messages')
        messages = np.zeros(np.size(messages_idx))

        sentiment = []
        dates = []
        for row in csv_reader:
            row = np.array(row)

            messages += np.array(row[messages_idx], dtype='float')
            sentiment.append(np.array(row[sent_idx], dtype='float64'))
            dates.append(row[0])

        top_idx = np.argsort(messages)[-top_num:]
        sentiment = np.array(sentiment)[:, top_idx]
        columns = columns[top_idx]

        start_idx = 0
        while start_idx < len(dates):
            if np.all(sentiment[start_idx] != 0.0):
                break
            start_idx+=1

        return DataFrame(data=sentiment[start_idx:, :],
                         index=to_datetime(dates[start_idx:]),
                         columns=columns)

