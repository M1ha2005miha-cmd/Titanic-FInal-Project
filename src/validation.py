from sklearn.model_selection import StratifiedKFold, cross_validate


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