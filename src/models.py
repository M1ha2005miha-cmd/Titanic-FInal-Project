from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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


def get_models(random_state):
    """Создает набор моделей для сравнения.

    Args:
        random_state: зерно случайности для моделей со случайностью внутри.

    Returns:
        Словарь: название модели - необученная модель sklearn.
    """
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "xgboost": XGBClassifier(random_state=random_state),
        "lightgbm": LGBMClassifier(random_state=random_state, verbose=-1)
    }
    
    return models