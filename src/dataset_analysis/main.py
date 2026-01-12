from .data_downloader import download_crypto_data
from .data_saver import save_data_to_local
from .market_analyzer import analyze_market_data
from .prediction_analyzer import run_all_predictions
from .price_plotter import plot_all_normalized_prices, plot_all_prices


def main() -> None:
    """Основная точка входа для анализа криптовалют."""

    default_cryptos = [
        ("Bitcoin", "BTC-USD"),
        ("Ethereum", "ETH-USD"),
        ("Solana", "SOL-USD"),
        ("XRP", "XRP-USD"),
        ("Cardano", "ADA-USD"),
    ]

    period = "3y"  # 3 года данных для 8 сегментов

    print("Начинаем анализ криптовалют...")

    data = download_crypto_data(default_cryptos, period)

    save_data_to_local(data)

    plot_all_prices(data)

    plot_all_normalized_prices(data)

    market_analysis = analyze_market_data(data)
    print("\nАнализ рынка:")
    print(market_analysis)

    prediction_results = run_all_predictions(data)
    print("\nРезультаты предсказаний:")
    print(prediction_results)

    print("\nАнализ завершен!")


if __name__ == "__main__":
    main()
