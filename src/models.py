from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def predict_by_sex(df):
    """Предсказывает выживание по полу.

    Args:
        df: таблица с пассажирами, в которой есть признак "Sex".

    Returns:
        Новый столбец предсказаний: 1 (выжил) для женщин, 0 (погиб) для мужчин.
    """

    predictions = (df["Sex"] == "female").astype(int)

    return predictions


def build_logistic_regression(preprocessor):
    """Создает модель логистической регрессии.

    Args:
        preprocessor: необученный ColumnTransformer,который обрабатывает числовые и категориальные признаки.

    Returns:
        Необученный Pipeline из preprocessor и LogisticRegression.
    """

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    return model
