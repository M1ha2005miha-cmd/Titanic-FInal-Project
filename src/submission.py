from numpy import False_
import pandas as pd 
from src.models import build_pipeline


def make_submission(model, preprocessor, X_train, y_train, X_test, passenger_ids, path):
    """Обучает модель на всем train и сохраняет предсказания для Kaggle.
     Args:
        model: необученная модель из словаря models.
        preprocessor: обработчик данных.
        X_train: признаки обучающей выборки.
        y_train: правильные ответы обучающей выборки.
        X_test: признаки тестовой выборки
        passenger_ids: столбец PassengerId из test.
        path: путь для сохранения CSV.

    Returns:
        Таблица с предсказаниями в формате Kaggle.
    """
    pipeline = build_pipeline(preprocessor, model)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions.astype(int),
    })
    submission.to_csv(path, index=False)
    return submission
