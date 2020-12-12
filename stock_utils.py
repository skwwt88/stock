import requests
import re
import pandas as pd
import datetime

stockids_training = [
    'sh600000',
    'sh600004',
    'sh600009',
    'sh600010',
    'sh600011',
    'sh600015',
    'sh600016',
    'sh600018',
    'sh600019',
    'sh600025',
    'sh600027',
    'sh600028',
    'sh600029',
    'sh600030',
    'sh600031',
    'sh600036',
    'sh600038',
    'sh600048',
    'sh600050',
    'sh600061',
    'sh600066',
    'sh600068',
    'sh600085',
    'sh600089',
    'sh600377',
    'sh601021',
    'sh601111',
    'sh601333'
]

def stock_kline_day(id: str, enrich = None, max_count = 30000):
    data_uri = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={0}&scale=240&ma=5&datalen={1}".format(id, max_count)
    response = requests.get(data_uri)
    stock_df = pd.read_json(response.content)
    stock_df['day'] = stock_df['day'].apply(pd.to_datetime)
    stock_df = stock_df.set_index('day')
    stock_df.drop(['ma_price5', 'ma_volume5'], inplace=True, axis=1)

    return stock_df if not enrich else enrich(stock_df, id)

def qfq(stock_df, id: str):
    def minusDay(date):
        dt = datetime.datetime.strptime(date, "%Y-%m-%d")
        dt = dt + datetime.timedelta(days=-1)
        return dt.strftime("%Y-%m-%d")

    qfq_uri = "https://finance.sina.com.cn/realstock/company/{0}/qfq.js".format(id)
    response = requests.get(qfq_uri)
    qfq_content = re.search(r'\[.*?\]', response.content.decode(encoding='utf-8')).group()
    qfq_df = pd.read_json(qfq_content)

    for index in range(1, len(qfq_df)):
        last_date = minusDay(qfq_df.iloc[-index - 1]['d'])
        start_date = qfq_df.iloc[-index]['d']
        f = qfq_df.iloc[-index]['f']

        stock_df.loc[start_date:last_date, 'open'] = stock_df.loc[start_date:last_date, 'open'] / f
        stock_df.loc[start_date:last_date, 'high'] = stock_df.loc[start_date:last_date, 'high'] / f
        stock_df.loc[start_date:last_date, 'low'] = stock_df.loc[start_date:last_date, 'low'] / f
        stock_df.loc[start_date:last_date, 'close'] = stock_df.loc[start_date:last_date, 'close'] / f
        stock_df.loc[start_date:last_date, 'volume'] = stock_df.loc[start_date:last_date, 'volume'] * f

        start_date = last_date

    return stock_df

if __name__ == "__main__":
    print(stock_kline_day('sh600029', qfq))