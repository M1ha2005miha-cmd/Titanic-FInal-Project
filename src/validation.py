from sklearn.model_selection import StratifiedKFold


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