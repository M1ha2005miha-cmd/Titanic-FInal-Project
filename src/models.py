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


def build_pipeline(preprocessor, model):
    """Соединить препроцессор и модель в один конвейер.

    Args:
        preprocessor: необученный ColumnTransformer,который обрабатывает числовые и категориальные признаки.
        model: необученная модель sklearn.

    Returns:
        Необученный Pipeline из preprocessor и model.
    """

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline