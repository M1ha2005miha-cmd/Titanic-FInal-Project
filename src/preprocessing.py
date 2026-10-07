from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_numeric_pipeline():
    """Создает конвейер обработки числовых признаков.

    Returns:
        Необученный Pipeline из SimpleImputer и StandardScaler.
    """

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return numeric_pipeline


def build_categorical_pipeline():
    """Создает конвейер обработки категориальных признаков.

    Returns:
        Необученный Pipeline из SimpleImputer и OneHotEncoder
    """

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return categorical_pipeline


def build_preprocessor(numeric_features, categorical_features):
    """Создает обработчик, который применяет свой конвейер к каждой группе признаков.

    Столбцы, не вошедшие ни в один список, отбрасываются.

    Args:
        numeric_features: список названий числовых столбцов.
        categorical_features: список названий категориальных столбцов.

    Returns:
        Необученный ColumnTransformer.
    """

    preprocessor = ColumnTransformer([
        ("numeric", build_numeric_pipeline(), numeric_features),
        ("categorical", build_categorical_pipeline(), categorical_features)
    ])

    return preprocessor