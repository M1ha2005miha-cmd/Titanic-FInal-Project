from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


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


class TitleAgeImputer(BaseEstimator, TransformerMixin):
    """Заполняет пропуски Age медианой возраста по титулу пассажира."""

    def fit(self, X, y=None):
        """Запоминает медиану Age для каждого Title и общую медиану по Age.

        Args:
            X: таблица признаков, в которой есть столбцы "Title" и "Age".
            y: не используется, нужен для совместимости со sklearn.
        
        Returns:
            Этот же объект, уже обученный.
        """
        self.medians_ = X.groupby("Title")["Age"].median()
        self.global_median_ = X["Age"].median()
        return self

    def transform(self, X):
        """Заполняте пропуски Age запомненными медианами.

        Args:
            X: таблица признаков, в которой есть столбцы "Title" и "Age".

        Returns:
            Копия таблицы без пропусков в Age.
        """
        X = X.copy()
        X["Age"] = X["Age"].fillna(X["Title"].map(self.medians_))
        X["Age"] = X["Age"].fillna(self.global_median_)
        return X


def build_preprocessor(numeric_features, categorical_features):
    """Создает обработчик, который заполняет Age по титулу, применяет свой конвейер к каждой группе признаков.

    Столбцы, не вошедшие ни в один список, отбрасываются.

    Args:
        numeric_features: список названий числовых столбцов.
        categorical_features: список названий категориальных столбцов.

    Returns:
        Необученный Pipeline: заполнение Age по Title, обработка числовых и категориальных признаков.
    """

    column_transformer = ColumnTransformer([
        ("numeric", build_numeric_pipeline(), numeric_features),
        ("categorical", build_categorical_pipeline(), categorical_features)
    ])

    preprocessor = Pipeline([
        ("age_imputer", TitleAgeImputer()),
        ("columns", column_transformer),
    ])

    return preprocessor