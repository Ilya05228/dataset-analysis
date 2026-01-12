import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict


def plot_all_prices(data_dict: Dict[str, pd.DataFrame]) -> None:
    """
    Создает один график со всеми ценами криптовалют.

    Args:
        data_dict: Словарь с данными криптовалют
    """
    os.makedirs("reports", exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 10))

    colors = ["blue", "green", "red", "purple", "orange"]

    for idx, (name, df) in enumerate(data_dict.items()):
        if len(df) > 0:
            ax.plot(
                df.index,
                df["Close"],
                linewidth=2,
                label=name,
                color=colors[idx % len(colors)],
            )

    ax.set_title("Цены криптовалют", fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel("Дата", fontsize=14)
    ax.set_ylabel("Цена (USD)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    filename = "reports/all_prices.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Создан график: {filename}")


def plot_all_normalized_prices(data_dict: Dict[str, pd.DataFrame]) -> None:
    """
    Создает один график со всеми нормализованными ценами криптовалют.

    Args:
        data_dict: Словарь с данными криптовалют
    """
    os.makedirs("reports", exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 10))

    colors = ["blue", "green", "red", "purple", "orange"]

    for idx, (name, df) in enumerate(data_dict.items()):
        if len(df) > 0:
            normalized_price = (df["Close"] - df["Close"].min()) / (
                df["Close"].max() - df["Close"].min()
            )
            ax.plot(
                df.index,
                normalized_price,
                linewidth=2,
                label=name,
                color=colors[idx % len(colors)],
            )

    ax.set_title(
        "Нормализованные цены криптовалют", fontsize=20, fontweight="bold", pad=20
    )
    ax.set_xlabel("Дата", fontsize=14)
    ax.set_ylabel("Нормализованная цена", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    filename = "reports/all_normalized_prices.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Создан график: {filename}")
