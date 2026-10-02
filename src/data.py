import pandas as pd


def load_data(train_path, test_path):
    """Загружает две таблицы пандас из файлов CSV.

    Args:
        train_path: путь к файлу с данными для обучения.
        test_path: путь к файлу с данными для теста.

    Returns:
        Таблицу пандас с обучающими данными.
        Таблицу пандас с тестовыми данными.
    """

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data