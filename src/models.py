def predict_by_sex(df):
    """Предсказывает выживание по полу.

    Args:
        df: таблица с пассажирами, в которой есть признак "Sex".

    Returns:
        Новый столбец предсказаний: 1 (выжил) для женщин, 0 (погиб) для мужчин.
    """

    predictions = (df["Sex"]=="female").astype(int)

    return predictions
