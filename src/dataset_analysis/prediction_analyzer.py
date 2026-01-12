import os
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


def prepare_prediction_data(
    df: pd.DataFrame, train_days: int = 30, test_days: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Подготавливает данные для предсказания.

    Args:
        df: DataFrame с данными
        train_days: Количество дней для обучения
        test_days: Количество дней для тестирования

    Returns:
        Кортеж с данными для обучения и тестирования
    """
    prices = df["Close"].values

    if len(prices) < train_days + test_days:
        train_days = len(prices) - test_days

    X_train = np.arange(train_days).reshape(-1, 1)
    y_train = prices[-train_days - test_days : -test_days]

    X_test = np.arange(train_days, train_days + test_days).reshape(-1, 1)
    y_test = prices[-test_days:]

    return X_train, y_train, X_test, y_test


def linear_regression_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """
    Предсказание с помощью линейной регрессии.

    Args:
        X_train: Обучающие признаки
        y_train: Обучающие метки
        X_test: Тестовые признаки

    Returns:
        Прогнозные значения
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def polynomial_regression_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, degree: int = 2
) -> np.ndarray:
    """
    Предсказание с помощью полиномиальной регрессии.

    Args:
        X_train: Обучающие признаки (временные метки)
        y_train: Обучающие метки
        X_test: Тестовые признаки (временные метки)
        degree: Степень полинома

    Returns:
        Прогнозные значения
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    return model.predict(X_test_poly)


def moving_average_predict(
    series: np.ndarray, train_size: int, test_size: int, window: int = 7
) -> np.ndarray:
    """
    Предсказание с помощью скользящего среднего.

    Args:
        series: Временной ряд
        train_size: Размер обучающей выборки
        test_size: Размер тестовой выборки
        window: Размер окна

    Returns:
        Прогнозные значения
    """
    predictions = []
    train_data = series[-train_size - test_size : -test_size]

    for i in range(test_size):
        # Используем только последние window значений из обучающей выборки + уже предсказанные
        if i < window:
            # Если у нас меньше window значений, используем все доступные
            available_data = (
                np.concatenate([train_data, predictions[:i]]) if i > 0 else train_data
            )
            window_start = max(0, len(available_data) - window)
        else:
            # Используем последние window предсказаний
            available_data = predictions[i - window : i]
            window_start = 0

        window_data = available_data[window_start:]
        predictions.append(np.mean(window_data))

    return np.array(predictions)


from statsmodels.tsa.holtwinters import ExponentialSmoothing


def exponential_smoothing_predict(
    series: np.ndarray,
    train_size: int,
    test_size: int,
    trend="additive",  # Для финансовых данных лучше additive
    seasonal_periods=None,
) -> np.ndarray:
    train = series[-train_size - test_size : -test_size]

    if len(train) < 2:
        return np.ones(test_size) * train.mean()

    try:
        # Проверяем, достаточно ли данных для сезонности
        if seasonal_periods and len(train) >= seasonal_periods * 2:
            seasonal = "additive"
        else:
            seasonal = None
            seasonal_periods = None

        model = ExponentialSmoothing(
            train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        )
        model_fit = model.fit(optimized=True)
        forecast = model_fit.forecast(test_size)
        return forecast
    except Exception as e:
        print(f"Exponential Smoothing error: {e}")
        # Fallback на простое скользящее среднее
        return np.ones(test_size) * train[-7:].mean()


def arima_predict(series: np.ndarray, train_size: int, test_size: int) -> np.ndarray:
    from pmdarima import auto_arima

    train = series[-train_size - test_size : -test_size]

    # Проверяем стационарность данных
    if len(train) < 10:
        raise ValueError("Not enough data for ARIMA")

    model = auto_arima(
        train,
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        max_d=2,
        seasonal=False,  # Для дневных данных криптовалют сезонность неочевидна
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        n_fits=10,
    )
    forecast = model.predict(n_periods=test_size)

    return forecast


def random_forest_predict(
    X_train_time: np.ndarray, y_train: np.ndarray, X_test_time: np.ndarray
) -> np.ndarray:
    """
    Предсказание с помощью случайного леса с лагами и техническими индикаторами.

    Args:
        X_train_time: Временные метки обучающей выборки
        y_train: Цены обучающей выборки
        X_test_time: Временные метки тестовой выборки

    Returns:
        Прогнозные значения
    """
    try:
        from sklearn.ensemble import RandomForestRegressor

        # Создаём расширенные признаки на основе истории цен
        def create_features(prices, time_indices=None):
            n = len(prices)
            features = []

            for i in range(n):
                feature_row = []

                # Лаги (1, 2, 3, 5, 7 дней)
                for lag in [1, 2, 3, 5, 7]:
                    if i >= lag:
                        feature_row.append(prices[i - lag])
                    else:
                        feature_row.append(prices[0])

                # Простое скользящее среднее (7 дней)
                if i >= 6:
                    feature_row.append(np.mean(prices[i - 6 : i + 1]))
                else:
                    feature_row.append(np.mean(prices[: i + 1]))

                # Волатильность (стандартное отклонение за 7 дней)
                if i >= 6:
                    feature_row.append(np.std(prices[i - 6 : i + 1]))
                else:
                    feature_row.append(np.std(prices[: i + 1]))

                # Минимум/максимум за 7 дней
                if i >= 6:
                    feature_row.append(np.min(prices[i - 6 : i + 1]))
                    feature_row.append(np.max(prices[i - 6 : i + 1]))
                else:
                    feature_row.append(np.min(prices[: i + 1]))
                    feature_row.append(np.max(prices[: i + 1]))

                # Добавляем временной признак (день)
                if time_indices is not None:
                    feature_row.append(time_indices[i])

                features.append(feature_row)

            return np.array(features)

        # Создаём признаки для обучающих данных
        X_train_features = create_features(y_train, X_train_time.flatten())

        # Для тестовых данных мы не можем использовать будущие цены, поэтому будем предсказывать по одному шагу
        # и обновлять историю для создания признаков следующего шага
        predictions = []
        history_prices = list(y_train)
        history_times = list(X_train_time.flatten())

        for test_time in X_test_time.flatten():
            # Создаём признаки на текущей истории
            current_features = create_features(
                np.array(history_prices), np.array(history_times)
            )[-1:]

            # Обучаем модель на всей текущей истории (можно кэшировать, но для простоты переобучаем)
            X_all = create_features(np.array(history_prices), np.array(history_times))
            y_all = history_prices

            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
            )
            model.fit(X_all, y_all)

            # Предсказываем следующий шаг
            pred = model.predict(current_features)[0]
            predictions.append(pred)

            # Обновляем историю для следующего шага
            history_prices.append(pred)
            history_times.append(test_time)

        return np.array(predictions)

    except Exception as e:
        print(f"Random Forest error: {e}")
        # Fallback на линейную регрессию
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X_train_time, y_train)
        return model.predict(X_test_time)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Рассчитывает все метрики оценки качества предсказаний.

    Args:
        y_true: Реальные значения
        y_pred: Предсказанные значения

    Returns:
        Словарь с метриками
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {}

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # R² score
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = np.nan

    # MAPE (Mean Absolute Percentage Error)
    try:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except:
        mape = np.nan

    # SMAPE (Symmetric Mean Absolute Percentage Error)
    try:
        smape = 100 * np.mean(
            2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
        )
    except:
        smape = np.nan

    # Максимальная ошибка
    max_error = np.max(np.abs(y_true - y_pred))

    # Медианная абсолютная ошибка
    median_ae = np.median(np.abs(y_true - y_pred))

    # Объясненная дисперсия
    explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
        "SMAPE": smape,
        "MaxError": max_error,
        "MedianAE": median_ae,
        "ExplainedVariance": explained_variance,
    }


def evaluate_method_on_segment(
    series: np.ndarray,
    dates: np.ndarray,
    start_idx: int,
    train_size: int = 30,
    test_size: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Оценивает все методы предсказания на одном сегменте данных.

    Args:
        series: Временной ряд цен
        dates: Массив дат
        start_idx: Начальный индекс сегмента
        train_size: Размер обучающей выборки
        test_size: Размер тестовой выборки

    Returns:
        Словарь с результатами для каждого метода
    """
    end_idx = start_idx + train_size + test_size
    if end_idx > len(series):
        return {}

    segment_series = series[start_idx:end_idx]
    segment_dates = dates[start_idx:end_idx]

    train = segment_series[:train_size]
    test = segment_series[train_size:]

    train_dates = segment_dates[:train_size]
    test_dates = segment_dates[train_size:]

    X_train_time = np.arange(train_size).reshape(-1, 1)
    X_test_time = np.arange(train_size, train_size + test_size).reshape(-1, 1)

    results = {}

    lr_pred = linear_regression_predict(X_train_time, train, X_test_time)
    metrics = calculate_all_metrics(test, lr_pred)
    results["linear_regression"] = {
        **metrics,
        "predictions": lr_pred,
        "train": train,
        "test": test,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "segment_start": start_idx,
        "segment_end": end_idx,
    }

    poly_pred = polynomial_regression_predict(
        X_train_time, train, X_test_time, degree=2
    )
    metrics = calculate_all_metrics(test, poly_pred)
    results["polynomial_regression"] = {
        **metrics,
        "predictions": poly_pred,
        "train": train,
        "test": test,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "segment_start": start_idx,
        "segment_end": end_idx,
    }

    ma_pred = moving_average_predict(segment_series, train_size, test_size, window=7)
    metrics = calculate_all_metrics(test, ma_pred)
    results["moving_average"] = {
        **metrics,
        "predictions": ma_pred,
        "train": train,
        "test": test,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "segment_start": start_idx,
        "segment_end": end_idx,
    }

    es_pred = exponential_smoothing_predict(
        segment_series,
        train_size,
        test_size,
    )
    metrics = calculate_all_metrics(test, es_pred)
    results["exponential_smoothing"] = {
        **metrics,
        "predictions": es_pred,
        "train": train,
        "test": test,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "segment_start": start_idx,
        "segment_end": end_idx,
    }

    arima_pred = arima_predict(segment_series, train_size, test_size)
    metrics = calculate_all_metrics(test, arima_pred)
    results["arima"] = {
        **metrics,
        "predictions": arima_pred,
        "train": train,
        "test": test,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "segment_start": start_idx,
        "segment_end": end_idx,
    }

    rf_pred = random_forest_predict(X_train_time, train, X_test_time)
    metrics = calculate_all_metrics(test, rf_pred)
    results["random_forest"] = {
        **metrics,
        "predictions": rf_pred,
        "train": train,
        "test": test,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "segment_start": start_idx,
        "segment_end": end_idx,
    }

    return results


def evaluate_all_methods_on_multiple_segments(
    series: np.ndarray,
    dates: np.ndarray,
    train_size: int = 30,
    test_size: int = 10,
    n_segments: int = 8,
) -> Dict[str, List[Dict]]:
    """
    Оценивает все методы предсказания на нескольких сегментах данных.

    Args:
        series: Временной ряд цен
        dates: Массив дат
        train_size: Размер обучающей выборки
        test_size: Размер тестовой выборки
        n_segments: Количество сегментов для анализа

    Returns:
        Словарь с результатами для каждого метода по всем сегментам
    """
    segment_size = train_size + test_size
    total_needed = segment_size * n_segments

    if len(series) < total_needed:
        print(f"  Недостаточно данных: нужно {total_needed} точек, есть {len(series)}")
        return {}

    results_by_method = {
        "linear_regression": [],
        "polynomial_regression": [],
        "moving_average": [],
        "exponential_smoothing": [],
        "arima": [],
        "random_forest": [],
    }

    # Берем последние данные для анализа (самые свежие)
    start_idx = len(series) - total_needed

    for segment_num in range(n_segments):
        current_start = start_idx + segment_num * segment_size
        segment_results = evaluate_method_on_segment(
            series, dates, current_start, train_size, test_size
        )

        if segment_results:
            for method_name, method_result in segment_results.items():
                method_result["segment_num"] = segment_num + 1
                method_result["start_date"] = dates[current_start]
                method_result["end_date"] = dates[current_start + segment_size - 1]
                results_by_method[method_name].append(method_result)

    return results_by_method


def plot_method_segments(
    crypto_name: str,
    method_name: str,
    best_segment: Dict,
    worst_segment: Dict,
    all_segments: List[Dict],
) -> None:
    """
    Создает графики для метода с лучшим и худшим сегментами.

    Args:
        crypto_name: Название криптовалюты
        method_name: Название метода
        best_segment: Лучший сегмент
        worst_segment: Худший сегмент
        all_segments: Все сегменты
    """
    method_names_ru = {
        "linear_regression": "Линейная регрессия",
        "polynomial_regression": "Полиномиальная регрессия",
        "moving_average": "Скользящее среднее",
        "exponential_smoothing": "Экспоненциальное сглаживание",
        "arima": "ARIMA",
        "random_forest": "Случайный лес",
    }

    method_name_ru = method_names_ru.get(method_name, method_name)

    # График 1: Лучший и худший сегменты
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Верхний график: лучший сегмент
    # Объединяем даты для плавного графика
    all_dates_best = np.concatenate(
        [best_segment["train_dates"], best_segment["test_dates"]]
    )
    all_values_best = np.concatenate([best_segment["train"], best_segment["test"]])

    # Создаем массив дат для предсказаний, начиная с последней обучающей даты
    # Добавляем точку соединения: последняя обучающая цена -> первое предсказание
    last_train_date = best_segment["train_dates"][-1]
    last_train_value = best_segment["train"][-1]
    first_pred_date = best_segment["test_dates"][0]
    first_pred_value = best_segment["predictions"][0]

    # Даты для предсказаний с точкой соединения
    pred_dates_with_connection = np.concatenate(
        [[last_train_date], best_segment["test_dates"]]
    )
    pred_values_with_connection = np.concatenate(
        [[last_train_value], best_segment["predictions"]]
    )

    ax1.plot(
        all_dates_best, all_values_best, "g-", linewidth=2, label="Реальные данные"
    )
    ax1.plot(
        best_segment["train_dates"],
        best_segment["train"],
        "b-",
        linewidth=3,
        label="Обучающая часть",
    )
    ax1.plot(
        pred_dates_with_connection,
        pred_values_with_connection,
        "r--",
        linewidth=2,
        label="Предсказания",
    )

    ax1.set_title(
        f"{method_name_ru}: {crypto_name}\nЛучший сегмент {best_segment['segment_num']} (RMSE={best_segment['RMSE']:.2f})",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xlabel("Даты", fontsize=16)
    ax1.set_ylabel("Цена", fontsize=16)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # Нижний график: худший сегмент
    # Объединяем даты для плавного графика
    all_dates_worst = np.concatenate(
        [worst_segment["train_dates"], worst_segment["test_dates"]]
    )
    all_values_worst = np.concatenate([worst_segment["train"], worst_segment["test"]])

    # Создаем массив дат для предсказаний, начиная с последней обучающей даты
    # Добавляем точку соединения: последняя обучающая цена -> первое предсказание
    last_train_date_worst = worst_segment["train_dates"][-1]
    last_train_value_worst = worst_segment["train"][-1]
    first_pred_date_worst = worst_segment["test_dates"][0]
    first_pred_value_worst = worst_segment["predictions"][0]

    # Даты для предсказаний с точкой соединения
    pred_dates_with_connection_worst = np.concatenate(
        [[last_train_date_worst], worst_segment["test_dates"]]
    )
    pred_values_with_connection_worst = np.concatenate(
        [[last_train_value_worst], worst_segment["predictions"]]
    )

    ax2.plot(
        all_dates_worst, all_values_worst, "g-", linewidth=2, label="Реальные данные"
    )
    ax2.plot(
        worst_segment["train_dates"],
        worst_segment["train"],
        "b-",
        linewidth=3,
        label="Обучающая часть",
    )
    ax2.plot(
        pred_dates_with_connection_worst,
        pred_values_with_connection_worst,
        "r--",
        linewidth=2,
        label="Предсказания",
    )

    ax2.set_title(
        f"{method_name_ru}: {crypto_name}\nХудший сегмент {worst_segment['segment_num']} (RMSE={worst_segment['RMSE']:.2f})",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xlabel("Даты", fontsize=16)
    ax2.set_ylabel("Цена", fontsize=16)
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    plt.suptitle(
        f"{method_name_ru}: {crypto_name} - Анализ {len(all_segments)} сегментов",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    filename = f"reports/predictions/{method_name}/{crypto_name.replace(' ', '_').lower()}_segments.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    # График 2: Сравнение RMSE по всем сегментам
    fig, ax = plt.subplots(figsize=(12, 8))

    segments = [s["segment_num"] for s in all_segments]
    rmses = [s["RMSE"] for s in all_segments]

    colors = [
        "green"
        if s == best_segment["segment_num"]
        else "red"
        if s == worst_segment["segment_num"]
        else "blue"
        for s in segments
    ]

    bars = ax.bar(segments, rmses, color=colors, alpha=0.7)

    # Подписи для лучшего и худшего
    ax.text(
        best_segment["segment_num"],
        best_segment["RMSE"] + max(rmses) * 0.02,
        "Лучший",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="green",
    )
    ax.text(
        worst_segment["segment_num"],
        worst_segment["RMSE"] + max(rmses) * 0.02,
        "Худший",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="red",
    )

    ax.set_title(
        f"{method_name_ru}: {crypto_name}\nRMSE по {len(all_segments)} сегментам",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Номер сегмента", fontsize=16)
    ax.set_ylabel("RMSE", fontsize=16)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    filename = f"reports/predictions/{method_name}/{crypto_name.replace(' ', '_').lower()}_rmse_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def run_all_predictions(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Запускает все методы предсказания для всех криптовалют.

    Args:
        data_dict: Словарь с данными криптовалют

    Returns:
        DataFrame с результатами анализа
    """
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    all_results = []
    train_size = 30
    test_size = 10
    n_segments = 8  # 8 сегментов для 2 лет данных

    method_folders = [
        "linear_regression",
        "polynomial_regression",
        "moving_average",
        "exponential_smoothing",
        "arima",
        "random_forest",
    ]

    for method_folder in method_folders:
        os.makedirs(f"reports/predictions/{method_folder}", exist_ok=True)

    for name, df in data_dict.items():
        print(f"\nАнализ {name}...")

        series = df["Close"].values
        dates = df.index.values

        # Анализируем на нескольких сегментах
        methods_results = evaluate_all_methods_on_multiple_segments(
            series, dates, train_size, test_size, n_segments
        )

        if not methods_results:
            print("  Недостаточно данных для анализа сегментов")
            continue

        # Для каждого метода находим лучший и худший сегмент
        for method_name, segment_results in methods_results.items():
            if not segment_results:
                continue

            # Находим лучший и худший сегмент по RMSE
            best_segment = min(segment_results, key=lambda x: x["RMSE"])
            worst_segment = max(segment_results, key=lambda x: x["RMSE"])

            # Сохраняем результаты в таблицу
            for segment in segment_results:
                result_row = {
                    "Криптовалюта": name,
                    "Метод": method_name,
                    "Сегмент": segment["segment_num"],
                    "Начало сегмента": segment["start_date"],
                    "Конец сегмента": segment["end_date"],
                    "MSE": segment.get("MSE", np.nan),
                    "RMSE": segment.get("RMSE", np.nan),
                    "MAE": segment.get("MAE", np.nan),
                    "R2": segment.get("R2", np.nan),
                    "MAPE": segment.get("MAPE", np.nan),
                    "SMAPE": segment.get("SMAPE", np.nan),
                    "MaxError": segment.get("MaxError", np.nan),
                    "MedianAE": segment.get("MedianAE", np.nan),
                    "ExplainedVariance": segment.get("ExplainedVariance", np.nan),
                    "Лучший сегмент": segment["segment_num"]
                    == best_segment["segment_num"],
                    "Худший сегмент": segment["segment_num"]
                    == worst_segment["segment_num"],
                }
                all_results.append(result_row)

            # Создаем графики для этого метода и этой криптовалюты
            plot_method_segments(
                name, method_name, best_segment, worst_segment, segment_results
            )

    # Сохраняем все результаты
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("reports/predictions_analysis.csv", index=False)

    print(f"\nСохранено {len(results_df)} записей в reports/predictions_analysis.csv")

    return results_df
