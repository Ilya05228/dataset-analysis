import os
import pandas as pd
from typing import Dict


def save_data_to_local(data_dict: Dict[str, pd.DataFrame]) -> None:
    """
    Сохраняет данные криптовалют в локальную папку.
    
    Args:
        data_dict: Словарь с данными криптовалют
    """
    os.makedirs("local_files", exist_ok=True)
    
    for name, df in data_dict.items():
        filename = f"local_files/{name.replace(' ', '_').lower()}_data.csv"
        df.to_csv(filename)
        print(f"Сохранено: {filename}")