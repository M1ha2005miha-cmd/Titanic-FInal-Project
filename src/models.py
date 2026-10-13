from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier

# Бустинг импортируется до torch т.к. просходить конфликт на macOS их копии OpenMP.
# Обратный порядок приводит к segmentation fault.
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import torch
from skorch.callbacks import EarlyStopping

from src.neural_net import TitanicNet, TitanicNetClassifier

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
    torch.manual_seed(random_state)
    torch.set_num_threads(1)
    
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "xgboost": XGBClassifier(random_state=random_state),
        "lightgbm": LGBMClassifier(random_state=random_state, verbose=-1),
        "neural_net": TitanicNetClassifier(
        TitanicNet,
        optimizer=torch.optim.Adam,
        lr=0.01,
        max_epochs=200,
        batch_size=32,
        callbacks=[EarlyStopping(patience=10)],
        verbose=0,
    ),
    }
    
    
    return models


def build_voting(models):
    """Собирает ансамбль мягкого голосования из моделей разных типов.

    Args:
        models: словарь моделей, в котором уже есть настроенные версии.

    Returns:
        VotingClassifier, усредняющий вероятности выживания.
    """
    return VotingClassifier(
        estimators=get_ensemble_estimators(models),
        voting="soft",
    )
    

def get_ensemble_estimators(models):
    """Возвращает модели разных типов для ансамблей.

    Args:
        models: словарь моделей, в котором уже есть настроенные версии.

    Returns:
        Список пар (имя, модель).
    """
    estimators=[
        ("logistic_regression", models["logistic_regression"]),
        ("knn", models["knn"]),
        ("random_forest", models["random_forest_tuned"]),
        ("lightgbm", models["lightgbm_tuned"]),
    ]
    return estimators


def build_stacking(models):
    """Собирает стекинг-ансамбль из моделей разных типов.

    Мета-модель (логистическая регрессия) учится на предсказаниях базовых моделей, сделаынных
    на отложенных фолдах, и подбирает, насколько доверять каждой из них.

    Args:
        models: словарь моделей, в котором уже есть настроенные версии.

    Returns:
        StackingClassifier с логистической регрессией в роли мета-модели.
    """
    return StackingClassifier(
        estimators=get_ensemble_estimators(models),
        final_estimator=LogisticRegression(),
    )