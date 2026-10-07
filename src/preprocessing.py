from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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

    categorial_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return categorial_pipeline