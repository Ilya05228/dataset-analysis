import pandas as pd
import numpy as np
from typing import Dict


def analyze_market_data(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Анализирует рыночные данные криптовалют.

    Args:
        data_dict: Словарь с данными криптовалют

    Returns:
        DataFrame с анализом рынка
    """
    analysis_results = []

    for name, df in data_dict.items():
        if len(df) == 0:
            continue

        total_volume = df["Volume"].sum()
        avg_daily_volume = df["Volume"].mean()

        daily_volatility = (df["High"] - df["Low"]) / df["Close"].shift(1)
        daily_volatility = daily_volatility.dropna()
        avg_daily_volatility = daily_volatility.mean()

        avg_price = df["Close"].mean()
        price_std = df["Close"].std()
        min_price = df["Close"].min()
        max_price = df["Close"].max()

        analysis_results.append(
            {
                "Криптовалюта": name,
                "Общий объем": total_volume,
                "Средний дневной объем": avg_daily_volume,
                "Средняя дневная волатильность": avg_daily_volatility,
                "Средняя цена": avg_price,
                "Мин цена": min_price,
                "Макс цена": max_price,
                "Стандартное отклонение": price_std,
                "Количество дней": len(df),
            }
        )

    analysis_df = pd.DataFrame(analysis_results)

    analysis_df.to_csv("reports/market_analysis.csv", index=False)

    return analysis_df
