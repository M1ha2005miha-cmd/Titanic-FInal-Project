import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.models import build_pipeline


def build_cv(n_splits, random_state):
    """Создает объект-разбиватель для стратифицированной кросс-валидации.

    Args:
        n_splits: на сколько частей делить данные.
        random_state: зерно случайности для воспроизводимого перемешивания.

    Returns:
        Объект StratifiedKFold.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return cv


def evaluate_model(model, X, y, cv, metrics):
    """Оценивает модель кросс-валидацией.

    Args:
        model: необученный конвейер.
        X: таблица признаков без целевого столбца.
        y: правильные ответы (1/0).
        cv: объект-разбиватель для кросс-валидации.
        metrics: список названий метрик.

    Returns:
        Словарь: для каждой метрики ключ test_<метрика> и значения по одному на каждый раунд.
    """
    scores = cross_validate(model, X, y, cv=cv, scoring=metrics)

    return scores


def summarize_scores(scores, metrics):
    """Считает среднее значение по раундам для каждой метрики.

    Args:
        scores: словарь результатов кросс-валидации с ключами test_<метрика>.
        metrics: список названий метрик.

    Returns:
        Словарь: название метрики - среднее значение.
    """
    summary = {}
    for metric in metrics:
        summary[metric] = scores[f"test_{metric}"].mean()

    return summary


def compare_models(models, preprocessor, X, y, cv, metrics):
    """Создает таблицу с данными по каждой модели для сравнения метрик.

    Args:
        models: словарь: название модели - необученная модель sklearn.
        preprocessor: обработчик данных.
        X: таблица признаков без целевого столбца.
        y: правильные ответы.
        cv: объект-разбиватель для кросс-валидации.
        metrics: список названий метрик.
    
    Returns:
        Таблица pandas: строки - модели, столбцы - метрики, значения - средние по раундам.
    """
    results = {}
    for name, model in models.items():
        pipeline = build_pipeline(preprocessor, model)
        scores = evaluate_model(pipeline, X, y, cv, metrics)
        results[name] = summarize_scores(scores, metrics)
    
    return pd.DataFrame(results).T