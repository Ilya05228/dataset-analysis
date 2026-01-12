import yfinance as yf
import pandas as pd
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta


def download_crypto_data(crypto_list: List[Tuple[str, str]], period: str = "1y") -> Dict[str, pd.DataFrame]:
    """
    Загружает данные о криптовалютах из Yahoo Finance.
    
    Args:
        crypto_list: Список кортежей (название, тикер)
        period: Период данных (например, "1y" для 1 года)
        
    Returns:
        Словарь с данными для каждой криптовалюты
    """
    data_dict = {}
    
    for name, ticker in crypto_list:
        print(f"Загрузка данных для {name} ({ticker})...")
        
        try:
            crypto = yf.Ticker(ticker)
            hist = crypto.history(period=period)
            
            if hist.empty:
                print(f"Нет данных для {name}")
                continue
            
            hist = hist.reset_index()
            hist['Date'] = pd.to_datetime(hist['Date'])
            hist.set_index('Date', inplace=True)
            
            data_dict[name] = hist
            print(f"Загружено {len(hist)} записей для {name}")
            
        except Exception as e:
            print(f"Ошибка при загрузке {name}: {e}")
    
    return data_dict