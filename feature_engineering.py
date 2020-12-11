

import pandas as pd
import numpy as np

from typing import List
from stock_utils import qfq, stock_kline_day

def full_feature_cols(featue_cols: List[str], steps: int) -> List[str]:
    result = []
    for step in range(1, steps + 1):
        for col in featue_cols:
            result.append("{0}_{1}".format(col, step))
    
    return result

def collect_stock_details(input: type(pd.DataFrame)) -> dict:
    result = {}
    for column in ['open', 'close', 'high', 'low']:
        std_column_name = "{0}_std".format(column)
        mean_column_name = "{0}_mean".format(column)

        prices = input[column] / input['close'].shift(1)
        prices = prices[prices < 1.11]
        prices = prices[prices > 0.89]

        result[mean_column_name] = prices.mean()
        result[std_column_name] = prices.std()

    log_volumes = np.log(input['volume'] + 1)
    result['volume_mean'] = log_volumes.mean()
    result['volume_std'] = log_volumes.std()

    return result

def load_data(stockids: List[str], p_steps: int = 22, n_steps: int = 2) -> type(pd.DataFrame): 
    result = []
    features_info = {}

    for stockid in stockids:
        stock_info = _load_single_stock_info(stockid)
        features_info[stockid] = collect_stock_details(stock_info)

        stock_info = _price_features_process(stock_info, features_info[stockid])
        stock_info = _volume_feature_process(stock_info, features_info[stockid])
        stock_info = _pivot_pre_data(stock_info, ['open_s', 'close_s', 'high_s', 'low_s', 'volume_s'], p_steps) #
        stock_info = _pivot_next_data(stock_info, ['high_s', 'low_s'], n_steps)
        stock_info = stock_info.dropna()
        result.append(stock_info)

    result = pd.concat(result)
    result = result.reset_index(drop=False)
    return result, features_info

def load_data_predict(stockids: List[str], features_info, p_steps: int = 22) -> type(pd.DataFrame): 
    feature_columns = ['open', 'close', 'high', 'low', 'volume']
    full_feature_columns = []
    for step in range(1, p_steps + 1):
        for col in feature_columns:
            full_feature_columns.append("p_{0}_s_{1}".format(col, step))

    predict_feature_data = pd.DataFrame(columns=full_feature_columns) 
    
    for stockid in stockids:
        stock_info = _load_single_stock_info(stockid)
        stock_info = _price_features_process(stock_info, features_info[stockid])
        stock_info = _volume_feature_process(stock_info, features_info[stockid])

        currentInfo = {}
        for column in feature_columns:
            mean_column_name = "{}_mean".format(column)
            std_column_name = "{}_std".format(column)
            for step in range(1, p_steps + 1):
                currentInfo["p_{0}_s_{1}".format(column, step)] = stock_info.iloc[-step]['{0}_s'.format(column)]
            
        currentInfo['stockid'] = stockid
        predict_feature_data = predict_feature_data.append(currentInfo, ignore_index=True)
    
    return predict_feature_data

def _pivot_pre_data(input: type(pd.DataFrame), columns: List[str], steps: int) -> type(pd.DataFrame):
    result = input.copy()
    for step in range(1, steps + 1):
        prev_step_data = input[columns].shift(step)  
        
        for column in columns:
            result["p_{0}_{1}".format(column, step)] = prev_step_data[column]

    return result

def _pivot_next_data(input: type(pd.DataFrame), columns: List[str], steps: int) -> type(pd.DataFrame):
    result = input.copy()
    for step in range(1, steps + 1):
        next_step_data = input[columns].shift(-step + 1)  
        
        for column in columns:
            result["n_{0}_{1}".format(column, step)] = next_step_data[column]

    return result

def _load_single_stock_info(stockid: str) -> type(pd.DataFrame): 
    result = stock_kline_day(stockid, qfq)
    result['stockid'] = stockid

    return result

def _price_features_process(input: type(pd.DataFrame), stats: dict):
    for column in ['open', 'close', 'high', 'low']:
        ratio_column_name = "{0}_ratio".format(column)
        input[ratio_column_name] = input[column] / input['close'].shift(1)
        input = input[input[ratio_column_name] < 1.11]
        input = input[input[ratio_column_name] > 0.89]

    for column in ['open', 'close', 'high', 'low']:
        std_column_name = "{0}_std".format(column)
        mean_column_name = "{0}_mean".format(column)
        standard_column_name = "{0}_s".format(column)

        input[std_column_name] = stats[std_column_name]
        input[mean_column_name] = stats[mean_column_name]
        input[standard_column_name] = (input[ratio_column_name] - input[mean_column_name]) / input[std_column_name]

    return input

def _volume_feature_process(input: type(pd.DataFrame), stats: dict): 
    column = 'volume'
    ln_column_name = "{0}_ln".format(column)
    std_column_name = "{0}_std".format(column)
    mean_column_name = "{0}_mean".format(column)
    standard_column_name = "{0}_s".format(column)

    input[ln_column_name] = np.log(input[column] + 1)
    input[std_column_name] = stats[std_column_name]
    input[mean_column_name] = stats[mean_column_name]
    input[standard_column_name] = (input[ln_column_name] - input[mean_column_name]) / input[std_column_name]

    return input

def _to_ratio_single_column(input: type(pd.DataFrame), column: str, rolling_step: int = 5) -> type(pd.DataFrame):
    base_column_name = "{0}_base".format(column)
    ratio_column_name = column + "_ratio"

    input[base_column_name] = input[column].rolling(rolling_step, min_periods=1).mean()
    input[ratio_column_name] = input[column] / input[base_column_name].shift(1)

def _standrad_single_column(input: type(pd.DataFrame), column: str, rolling_step: int = 66) -> type(pd.DataFrame):
    ratio_column_name = column + "_ratio"
    std_column_name = "{0}_s_std".format(column)
    s_mean_column_name = "{0}_s_mean".format(column)
    standard_column_name = column + "_s"

    input[std_column_name] = input[ratio_column_name].rolling(rolling_step, min_periods=1).std()
    input[s_mean_column_name] = input[ratio_column_name].rolling(rolling_step, min_periods=1).mean()
    input[standard_column_name] = (input[ratio_column_name] - input[s_mean_column_name].shift(1)) / input[std_column_name].shift(1)


if __name__ == "__main__":
    data, features_info = load_data(['sh600029'])
    p_data = load_data_predict(['sh600029'], features_info)

    print(data[['high_s', 'n_high_s_1']].describe())
    print(p_data[['p_high_s_1', 'p_high_s_22']].describe())