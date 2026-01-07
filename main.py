import shutil
import warnings
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# Настройки стиля графиков
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def clean_output_and_check_dataset():
    """Очистка папки output и проверка что dataset не пустой"""
    print("Проверка и подготовка директорий...")

    dataset_path = Path("./files/dataset")
    output_path = Path("./files/output")

    # Создаем директории если их нет
    dataset_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # Очищаем папку output
    if output_path.exists():
        for item in output_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"Папка output очищена: {output_path}")

    # Проверяем что dataset не пустой
    crypto_files = list(dataset_path.glob("*.csv"))
    if len(crypto_files) == 0:
        print("ОШИБКА: Папка files/dataset пуста!")
        print("Пожалуйста, добавьте CSV файлы криптовалют в папку files/dataset/")
        print("Пример файлов: Bitcoin-USD.csv, Ethereum-USD.csv и т.д.")
        return False

    print(f"Найдено {len(crypto_files)} файлов криптовалют в dataset")
    return True


def download_dataset():
    """Простое скачивание и копирование всего датасета криптовалют"""
    print("Скачивание датасета криптовалют...")
    dataset_path = Path("./files/dataset")
    output_path = Path("./files/output")

    dataset_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    return dataset_path, output_path


def load_crypto_data(dataset_path):
    """Загрузка всех CSV файлов криптовалют"""
    crypto_files = list(dataset_path.glob("*.csv"))
    print(f"Найдено {len(crypto_files)} файлов криптовалют")

    crypto_data = {}
    for file in crypto_files:
        try:
            # Пропускаем вторую строку (единицы измерения)
            df = pd.read_csv(file, skiprows=[1])
            # Извлекаем название криптовалюты из имени файла
            crypto_name = file.stem.replace("-USD", "").replace("_", " ").title()
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            crypto_data[crypto_name] = df
        except Exception as e:
            print(f"Ошибка при загрузке {file}: {e}")

    return crypto_data


def calculate_annual_volume(crypto_data):
    """Расчет годового объема торгов для каждой криптовалюты"""
    annual_volumes = {}
    for name, df in crypto_data.items():
        # Берем данные за последний год
        last_year = df.index.max() - timedelta(days=365)
        df_last_year = df[df.index >= last_year]

        if not df_last_year.empty:
            annual_volume = df_last_year["Volume"].sum()
            annual_volumes[name] = annual_volume

    # Сортируем по объему и берем топ-10
    top_10 = dict(sorted(annual_volumes.items(), key=lambda x: x[1], reverse=True)[:10])

    return top_10


def plot_price_trends(crypto_data, top_10_cryptos, output_path):
    """График цен топ-10 криптовалют за последний год"""
    print("Создание графиков цен...")

    # График 1: Цены без нормализации
    plt.figure(figsize=(15, 8))
    for crypto in top_10_cryptos:
        df = crypto_data[crypto]
        last_year = df.index.max() - timedelta(days=365)
        df_last_year = df[df.index >= last_year]

        if not df_last_year.empty:
            plt.plot(
                df_last_year.index, df_last_year["Close"], label=crypto, linewidth=2
            )

    plt.title(
        "Цены топ-10 криптовалют за последний год", fontsize=16, fontweight="bold"
    )
    plt.xlabel("Дата", fontsize=12)
    plt.ylabel("Цена закрытия (USD)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "price_trends_actual.png", dpi=300, bbox_inches="tight")
    plt.close()

    # График 2: Нормализованные цены
    plt.figure(figsize=(15, 8))
    for crypto in top_10_cryptos:
        df = crypto_data[crypto]
        last_year = df.index.max() - timedelta(days=365)
        df_last_year = df[df.index >= last_year]

        if not df_last_year.empty:
            normalized = (df_last_year["Close"] - df_last_year["Close"].min()) / (
                df_last_year["Close"].max() - df_last_year["Close"].min()
            )
            plt.plot(df_last_year.index, normalized, label=crypto, linewidth=2)

    plt.title(
        "Нормализованные цены топ-10 криптовалют за последний год",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Дата", fontsize=12)
    plt.ylabel("Нормализованная цена (0-1)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_path / "price_trends_normalized.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("Графики цен сохранены")


def plot_volume_bars(top_10_volumes, output_path):
    """График объемов торгов топ-10 криптовалют"""
    print("Создание графика объемов...")

    plt.figure(figsize=(12, 8))

    cryptos = list(top_10_volumes.keys())
    volumes = list(top_10_volumes.values())

    # Конвертируем объемы в миллиарды для лучшей читаемости
    volumes_billions = [v / 1e9 for v in volumes]

    bars = plt.barh(
        cryptos, volumes_billions, color=plt.cm.viridis(np.linspace(0, 1, len(cryptos)))
    )

    plt.xlabel("Годовой объем торгов (млрд USD)", fontsize=12)
    plt.title(
        "Топ-10 криптовалют по годовому объему торгов", fontsize=16, fontweight="bold"
    )
    plt.grid(True, alpha=0.3, axis="x")

    # Добавляем значения на столбцы
    for bar, volume in zip(bars, volumes_billions):
        plt.text(
            bar.get_width() + max(volumes_billions) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{volume:,.1f}B",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path / "volume_bars.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("График объемов сохранен")


def plot_total_volume_analysis(crypto_data, output_path):
    """Анализ суммарного объема торгов"""
    print("Анализ суммарного объема торгов...")

    # Суммарный объем по всем криптовалютам
    total_volume_by_date = pd.Series(dtype=float)

    for name, df in crypto_data.items():
        # Берем данные за последний год
        last_year = df.index.max() - timedelta(days=365)
        df_last_year = df[df.index >= last_year]

        if not df_last_year.empty:
            if total_volume_by_date.empty:
                total_volume_by_date = df_last_year["Volume"]
            else:
                # Выравниваем по датам
                total_volume_by_date = total_volume_by_date.add(
                    df_last_year["Volume"], fill_value=0
                )

    if not total_volume_by_date.empty:
        # Ограничиваем данные только последним годом (2025-2026)
        # Находим минимальную и максимальную дату
        min_date = total_volume_by_date.index.min()
        max_date = total_volume_by_date.index.max()

        # Если данные охватывают больше года, ограничиваем последним годом
        if (max_date - min_date).days > 365:
            one_year_ago = max_date - timedelta(days=365)
            total_volume_by_date = total_volume_by_date[
                total_volume_by_date.index >= one_year_ago
            ]

        # График суммарного объема по дням
        plt.figure(figsize=(15, 8))
        plt.plot(total_volume_by_date.index, total_volume_by_date / 1e9, linewidth=2)
        plt.title(
            "Суммарный дневной объем торгов всех криптовалют",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Дата", fontsize=12)
        plt.ylabel("Объем торгов (млрд USD)", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Форматируем ось X для отображения только последнего года
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(
            output_path / "total_volume_daily.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # График суммарного объема по месяцам
        monthly_volume = total_volume_by_date.resample("M").sum()
        plt.figure(figsize=(12, 8))

        # Создаем список месяцев для отображения
        months = [d.strftime("%Y-%m") for d in monthly_volume.index]

        plt.bar(
            range(len(monthly_volume)),
            monthly_volume.values / 1e9,
            color=plt.cm.coolwarm(np.linspace(0, 1, len(monthly_volume))),
        )
        plt.title(
            "Суммарный месячный объем торгов всех криптовалют",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Месяц", fontsize=12)
        plt.ylabel("Объем торгов (млрд USD)", fontsize=12)

        # Показываем только каждый 2-й месяц для читаемости
        if len(months) > 6:
            step = max(1, len(months) // 6)
            plt.xticks(
                range(0, len(months), step),
                [months[i] for i in range(0, len(months), step)],
                rotation=45,
            )
        else:
            plt.xticks(range(len(months)), months, rotation=45)

        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            output_path / "total_volume_monthly.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("Графики суммарного объема сохранены")

    return total_volume_by_date


def calculate_volatility(crypto_data, top_10_cryptos):
    """Расчет дневной волатильности"""
    print("Расчет волатильности...")

    volatility_data = {}
    for crypto in top_10_cryptos:
        df = crypto_data[crypto]
        last_year = df.index.max() - timedelta(days=365)
        df_last_year = df[df.index >= last_year]

        if not df_last_year.empty:
            # Дневная волатильность как (High - Low) / Close
            df_last_year["Daily_Volatility"] = (
                df_last_year["High"] - df_last_year["Low"]
            ) / df_last_year["Close"]
            volatility_data[crypto] = df_last_year["Daily_Volatility"]

    # Создаем DataFrame для волатильности
    volatility_df = pd.DataFrame(volatility_data)

    return volatility_df


def plot_volatility_clusters(volatility_df, output_path):
    """Кластеризация по волатильности"""
    print("Кластеризация по волатильности...")

    # Средняя годовая волатильность для каждой криптовалюты
    avg_volatility = volatility_df.mean()

    # График 1: Средняя волатильность
    plt.figure(figsize=(12, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(avg_volatility)))
    bars = plt.bar(
        range(len(avg_volatility)), avg_volatility.values * 100, color=colors
    )

    plt.xlabel("Криптовалюты", fontsize=12)
    plt.ylabel("Средняя дневная волатильность (%)", fontsize=12)
    plt.title(
        "Средняя дневная волатильность топ-10 криптовалют",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(
        range(len(avg_volatility)), avg_volatility.index, rotation=45, ha="right"
    )
    plt.grid(True, alpha=0.3, axis="y")

    # Добавляем значения
    for bar, value in zip(bars, avg_volatility.values * 100):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path / "volatility_bars.png", dpi=300, bbox_inches="tight")
    plt.close()

    # График 2: Дендрограмма кластеризации
    plt.figure(figsize=(12, 8))

    # Используем корреляционную матрицу для кластеризации
    corr_matrix = volatility_df.corr()

    # Преобразуем корреляцию в расстояние
    distance_matrix = 1 - corr_matrix
    condensed_dist = squareform(distance_matrix)

    # Иерархическая кластеризация
    linkage_matrix = linkage(condensed_dist, method="ward")

    dendrogram(linkage_matrix, labels=volatility_df.columns, leaf_rotation=90)

    plt.title(
        "Кластеризация криптовалют по волатильности", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Криптовалюты", fontsize=12)
    plt.ylabel("Расстояние", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path / "volatility_dendrogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Графики кластеризации сохранены")

    return avg_volatility


def plot_correlation_heatmaps(crypto_data, top_10_cryptos, output_path):
    """Корреляционные матрицы цен и волатильности"""
    print("Создание корреляционных матриц...")

    # Подготовка данных для корреляции цен
    price_data = {}
    volatility_data = {}

    for crypto in top_10_cryptos:
        df = crypto_data[crypto]
        last_year = df.index.max() - timedelta(days=365)
        df_last_year = df[df.index >= last_year]

        if not df_last_year.empty:
            price_data[crypto] = df_last_year["Close"]
            # Рассчитываем дневную волатильность
            daily_vol = (df_last_year["High"] - df_last_year["Low"]) / df_last_year[
                "Close"
            ]
            volatility_data[crypto] = daily_vol

    price_df = pd.DataFrame(price_data)
    volatility_df = pd.DataFrame(volatility_data)

    # Корреляция цен
    price_corr = price_df.corr()

    # Корреляция волатильности
    vol_corr = volatility_df.corr()

    # Heatmap 1: Корреляция цен
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        price_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Корреляция цен закрытия топ-10 криптовалют\n(последний год)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path / "correlation_prices.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Heatmap 2: Корреляция волатильности
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        vol_corr,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Корреляция волатильности топ-10 криптовалют\n(последний год)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        output_path / "correlation_volatility.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("Корреляционные матрицы сохранены")

    # Находим топ-5 самых коррелированных пар для цен
    price_corr_matrix = price_corr.values
    np.fill_diagonal(price_corr_matrix, 0)  # Игнорируем диагональ

    # Получаем топ-5 пар
    price_top_pairs = []
    for i in range(len(price_corr.columns)):
        for j in range(i + 1, len(price_corr.columns)):
            price_top_pairs.append(
                (price_corr.columns[i], price_corr.columns[j], price_corr.iloc[i, j])
            )

    price_top_5 = sorted(price_top_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]

    # Находим топ-5 самых коррелированных пар для волатильности
    vol_corr_matrix = vol_corr.values
    np.fill_diagonal(vol_corr_matrix, 0)

    vol_top_pairs = []
    for i in range(len(vol_corr.columns)):
        for j in range(i + 1, len(vol_corr.columns)):
            vol_top_pairs.append(
                (vol_corr.columns[i], vol_corr.columns[j], vol_corr.iloc[i, j])
            )

    vol_top_5 = sorted(vol_top_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]

    return price_top_5, vol_top_5, price_corr, vol_corr


def linear_regression_analysis(crypto_data, top_10_cryptos, output_path, top=5):
    """Линейная регрессия для прогнозирования цен на 30 дней вперед"""
    print("Анализ линейной регрессии...")

    # Берем топ-2 криптовалюты по объему
    top_2 = list(top_10_cryptos.keys())[:top]

    results = {}

    for crypto in top_2:
        df = crypto_data[crypto]

        # Берем последние 60 дней для обучения
        last_date = df.index.max()
        train_start = last_date - timedelta(
            days=90
        )  # 60 дней обучения + 30 дней тестирования

        df_train = df[df.index >= train_start].copy()

        if len(df_train) < 60:
            print(f"Недостаточно данных для {crypto}")
            continue

        # Создаем признаки: номер дня
        df_train["Day"] = range(len(df_train))

        # Разделяем на обучение (первые 60 дней) и тест (последние 30 дней)
        X_train = df_train["Day"].values[:60].reshape(-1, 1)
        y_train = df_train["Close"].values[:60]

        X_test = df_train["Day"].values[60:].reshape(-1, 1)
        y_test = df_train["Close"].values[60:]

        # Обучаем модель
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Прогнозируем
        y_pred = model.predict(X_test)

        # Оцениваем точность
        mae = np.mean(np.abs(y_pred - y_test))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = model.score(X_test, y_test)

        results[crypto] = {
            "model": model,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "actual": y_test,
            "predicted": y_pred,
            "dates": df_train.index[60:],
        }

        # Визуализация
        fig, ax = plt.subplots(figsize=(12, 6))

        # Фактические данные
        ax.plot(
            df_train.index[:60], y_train, "b-", label="Данные для обучения", linewidth=2
        )
        ax.plot(
            df_train.index[60:], y_test, "g-", label="Фактические значения", linewidth=2
        )
        ax.plot(df_train.index[60:], y_pred, "r--", label="Прогноз", linewidth=2)

        ax.set_title(
            f"Линейная регрессия: {crypto}\nПрогноз на 30 дней вперед",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Дата", fontsize=12)
        ax.set_ylabel("Цена (USD)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Добавляем метрики
        textstr = f"MAE: ${mae:.2f}\nMAPE: {mape:.1f}%\nR²: {r2:.3f}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()
        plt.savefig(
            output_path / f"regression_{crypto}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    print("Анализ регрессии завершен")
    return results


def calculate_max_drawdown_gain(crypto_data, top_10_cryptos):
    """Расчет максимальных просадок и роста"""
    print("Расчет максимальных просадок и роста...")

    results = {}

    for crypto in list(top_10_cryptos.keys())[:5]:  # Топ-5
        df = crypto_data[crypto]

        if len(df) < 30:  # Нужен хотя бы месяц данных
            continue

        # Максимальная месячная просадка (от хая до лоя)
        df["Monthly_High"] = df["High"].rolling(window=30).max()
        df["Monthly_Low"] = df["Low"].rolling(window=30).min()

        # Убираем NaN
        df_valid = df.dropna(subset=["Monthly_High", "Monthly_Low"])

        if not df_valid.empty:
            max_drawdown = (
                (df_valid["Monthly_Low"] - df_valid["Monthly_High"])
                / df_valid["Monthly_High"]
            ).min() * 100

            max_gain = (
                (df_valid["Monthly_High"] - df_valid["Monthly_Low"])
                / df_valid["Monthly_Low"]
            ).max() * 100

            results[crypto] = {
                "max_drawdown_pct": max_drawdown,
                "max_gain_pct": max_gain,
                "max_drawdown_date": df_valid.index[
                    (
                        (df_valid["Monthly_Low"] - df_valid["Monthly_High"])
                        / df_valid["Monthly_High"]
                    ).argmin()
                ],
                "max_gain_date": df_valid.index[
                    (
                        (df_valid["Monthly_High"] - df_valid["Monthly_Low"])
                        / df_valid["Monthly_Low"]
                    ).argmax()
                ],
            }

    return results


def calculate_returns(crypto_data, top_10_cryptos):
    """Расчет доходности за разные периоды"""
    print("Расчет доходности...")

    returns_data = {}

    for crypto in list(top_10_cryptos.keys())[:5]:  # Топ-5
        df = crypto_data[crypto]

        if len(df) < 365 * 5:  # Нужны данные за 5 лет
            continue

        # Доходность за 1 год
        if len(df) >= 365:
            year_ago = df.index.max() - timedelta(days=365)
            df_year = df[df.index >= year_ago]
            if not df_year.empty:
                return_1y = (
                    df_year["Close"].iloc[-1] / df_year["Close"].iloc[0] - 1
                ) * 100
            else:
                return_1y = np.nan
        else:
            return_1y = np.nan

        # Доходность за 2 года
        if len(df) >= 365 * 2:
            two_years_ago = df.index.max() - timedelta(days=365 * 2)
            df_2y = df[df.index >= two_years_ago]
            if not df_2y.empty:
                return_2y = (df_2y["Close"].iloc[-1] / df_2y["Close"].iloc[0] - 1) * 100
            else:
                return_2y = np.nan
        else:
            return_2y = np.nan

        # Доходность за 5 лет
        if len(df) >= 365 * 5:
            five_years_ago = df.index.max() - timedelta(days=365 * 5)
            df_5y = df[df.index >= five_years_ago]
            if not df_5y.empty:
                return_5y = (df_5y["Close"].iloc[-1] / df_5y["Close"].iloc[0] - 1) * 100
            else:
                return_5y = np.nan
        else:
            return_5y = np.nan

        returns_data[crypto] = {
            "1_year_return_pct": return_1y,
            "2_year_return_pct": return_2y,
            "5_year_return_pct": return_5y,
        }

    return returns_data


def save_separate_tables(results, output_path):
    """Сохранение отдельных таблиц в CSV формате"""

    # Таблица 1: Топ-10 по объему
    if "top_10_volumes" in results:
        df_volume = pd.DataFrame(
            [
                {
                    "№": i,
                    "Криптовалюта": crypto,
                    "Годовой объем (млрд USD)": f"${volume / 1e9:,.1f}",
                }
                for i, (crypto, volume) in enumerate(
                    results["top_10_volumes"].items(), 1
                )
            ]
        )
        df_volume.to_csv(
            output_path / "table_top10_volume.csv", index=False, encoding="utf-8"
        )

    # Таблица 2: Волатильность
    if "avg_volatility" in results:
        df_volatility = pd.DataFrame(
            [
                {"Криптовалюта": crypto, "Волатильность (%)": f"{vol * 100:.2f}%"}
                for crypto, vol in results["avg_volatility"].items()
            ]
        )
        df_volatility.to_csv(
            output_path / "table_volatility.csv", index=False, encoding="utf-8"
        )

    # Таблица 3: Корреляция цен
    if "price_top_5" in results:
        df_corr_prices = pd.DataFrame(
            [
                {"№": i, "Пара": f"{crypto1} - {crypto2}", "Корреляция": f"{corr:.3f}"}
                for i, (crypto1, crypto2, corr) in enumerate(results["price_top_5"], 1)
            ]
        )
        df_corr_prices.to_csv(
            output_path / "table_correlation_prices.csv", index=False, encoding="utf-8"
        )

    # Таблица 4: Корреляция волатильности
    if "vol_top_5" in results:
        df_corr_vol = pd.DataFrame(
            [
                {"№": i, "Пара": f"{crypto1} - {crypto2}", "Корреляция": f"{corr:.3f}"}
                for i, (crypto1, crypto2, corr) in enumerate(results["vol_top_5"], 1)
            ]
        )
        df_corr_vol.to_csv(
            output_path / "table_correlation_volatility.csv",
            index=False,
            encoding="utf-8",
        )

    # Таблица 5: Регрессия
    if "regression_results" in results:
        df_regression = pd.DataFrame(
            [
                {
                    "Криптовалюта": crypto,
                    "MAE (USD)": f"${res['mae']:.2f}",
                    "MAPE (%)": f"{res['mape']:.1f}%",
                    "R²": f"{res['r2']:.3f}",
                }
                for crypto, res in results["regression_results"].items()
            ]
        )
        df_regression.to_csv(
            output_path / "table_regression.csv", index=False, encoding="utf-8"
        )

    # Таблица 6: Просадки и рост
    if "max_drawdown_gain" in results:
        df_drawdown = pd.DataFrame(
            [
                {
                    "Криптовалюта": crypto,
                    "Макс. просадка (%)": f"{res['max_drawdown_pct']:.1f}%",
                    "Макс. рост (%)": f"{res['max_gain_pct']:.1f}%",
                }
                for crypto, res in results["max_drawdown_gain"].items()
            ]
        )
        df_drawdown.to_csv(
            output_path / "table_drawdown_gain.csv", index=False, encoding="utf-8"
        )

    # Таблица 7: Доходность
    if "returns" in results:
        returns_data = []
        for crypto, res in results["returns"].items():
            row = {"Криптовалюта": crypto}
            if not np.isnan(res["1_year_return_pct"]):
                row["1 год (%)"] = f"{res['1_year_return_pct']:.1f}%"
            else:
                row["1 год (%)"] = "-"
            if not np.isnan(res["2_year_return_pct"]):
                row["2 года (%)"] = f"{res['2_year_return_pct']:.1f}%"
            else:
                row["2 года (%)"] = "-"
            if not np.isnan(res["5_year_return_pct"]):
                row["5 лет (%)"] = f"{res['5_year_return_pct']:.1f}%"
            else:
                row["5 лет (%)"] = "-"
            returns_data.append(row)

        df_returns = pd.DataFrame(returns_data)
        df_returns.to_csv(
            output_path / "table_returns.csv", index=False, encoding="utf-8"
        )

    print("Отдельные таблицы сохранены в CSV файлы")


def main():
    """Основная функция анализа"""
    print("Запуск анализа криптовалют...")

    # Очищаем output и проверяем dataset
    if not clean_output_and_check_dataset():
        return

    dataset_path = Path("./files/dataset")
    output_path = Path("./files/output")

    # Загружаем данные
    crypto_data = load_crypto_data(dataset_path)

    if not crypto_data:
        print("Ошибка: не удалось загрузить данные криптовалют")
        return

    # Рассчитываем топ-10 по годовому объему
    top_10_volumes = calculate_annual_volume(crypto_data)
    print("\nТоп-10 криптовалют по годовому объему:")
    for i, (crypto, volume) in enumerate(top_10_volumes.items(), 1):
        print(f"{i}. {crypto}: ${volume / 1e9:.1f}B")

    # Создаем графики
    plot_price_trends(crypto_data, top_10_volumes.keys(), output_path)
    plot_volume_bars(top_10_volumes, output_path)

    # Анализ суммарного объема
    plot_total_volume_analysis(crypto_data, output_path)

    # Анализ волатильности
    volatility_df = calculate_volatility(crypto_data, top_10_volumes.keys())
    avg_volatility = plot_volatility_clusters(volatility_df, output_path)

    # Корреляционный анализ
    price_top_5, vol_top_5, price_corr, vol_corr = plot_correlation_heatmaps(
        crypto_data, top_10_volumes.keys(), output_path
    )

    # Линейная регрессия
    regression_results = linear_regression_analysis(
        crypto_data, top_10_volumes, output_path
    )

    # Максимальные просадки и рост
    max_drawdown_gain = calculate_max_drawdown_gain(crypto_data, top_10_volumes)

    # Расчет доходности
    returns = calculate_returns(crypto_data, top_10_volumes)
    # Собираем все результаты
    results = {
        "top_10_volumes": top_10_volumes,
        "avg_volatility": avg_volatility,
        "price_top_5": price_top_5,
        "vol_top_5": vol_top_5,
        "price_correlation": price_corr,
        "volatility_correlation": vol_corr,
        "regression_results": regression_results,
        "max_drawdown_gain": max_drawdown_gain,
        "returns": returns,
    }

    # Сохраняем отдельные таблицы
    save_separate_tables(results, output_path)

    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print(f"Все результаты сохранены в: {output_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
