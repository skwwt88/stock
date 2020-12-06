

import pandas as pd

from typing import List
from stock_utils import qfq, stock_kline_day


def load_data(stockids: List[str], p_steps: int = 22, n_steps: int = 3) -> type(pd.DataFrame): 
    result = []

    for stockid in stockids:
        stock_info = _load_single_stock_info(stockid)
        stock_info = standrad(stock_info)
        stock_info = pivot_pre_data(stock_info, ['open_s', 'close_s', 'high_s', 'low_s', 'volume_s'], p_steps)
        stock_info = pivot_next_data(stock_info, ['high_s', 'low_s'], n_steps)
        stock_info = stock_info.dropna()
        result.append(stock_info)

    result = pd.concat(result)
    result = result.reset_index(drop=False)
    return result

def standrad(input: type(pd.DataFrame)) -> type(pd.DataFrame):
    result = input.copy()

    _to_ratio_single_column(result, 'open')
    _standrad_single_column(result, 'open')
    _to_ratio_single_column(result, 'close')
    _standrad_single_column(result, 'close')
    _to_ratio_single_column(result, 'low')
    _standrad_single_column(result, 'low')
    _to_ratio_single_column(result, 'high')
    _standrad_single_column(result, 'high')
    _to_ratio_single_column(result, 'volume')
    _standrad_single_column(result, 'volume')
    
    return result

def pivot_pre_data(input: type(pd.DataFrame), columns: List[str], steps: int) -> type(pd.DataFrame):
    result = input.copy()
    for step in range(1, steps + 1):
        prev_step_data = input[columns].shift(step)  
        
        for column in columns:
            result["p_{0}_{1}".format(column, step)] = prev_step_data[column]

    return result

def pivot_next_data(input: type(pd.DataFrame), columns: List[str], steps: int) -> type(pd.DataFrame):
    result = input.copy()
    for step in range(1, steps + 1):
        next_step_data = input[columns].shift(-step)  
        
        for column in columns:
            result["n_{0}_{1}".format(column, step)] = next_step_data[column]

    return result

def _load_single_stock_info(stockid: str) -> type(pd.DataFrame): 
    result = stock_kline_day(stockid, qfq)
    result['stockid'] = stockid

    return result

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
    data = load_data(['sh600029', 'sh601595'])

    print(data)