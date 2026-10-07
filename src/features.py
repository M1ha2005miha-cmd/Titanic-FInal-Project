import pandas as pd

def add_title(df):
    """ Добавляет новый признак "Title" - титул пассажира из столбца "Name".
    Редкие титулы объединяются в категорию "Rare".

    Args:
        df: таблица с данными пассажиров, в которой есть столбец "Name"

    Returns:
        Копия таблицы с новым признаком "Title"
    """

    df = df.copy()
    main_titles = ["Mr", "Mrs", "Miss", "Master"]
    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    }
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    df["Title"] = df["Title"].replace(title_map)
    df["Title"] = df["Title"].where(df["Title"].isin(main_titles), "Rare")

    return df


def add_family_features(df):
    """Добавляет признаки "FamilySize" и "FamilyGroup" - размер семьи и его группу. 

    Args:
        df: таблица с данными пассажиров в которой есть "SibSp" и "Parch"

    Returns:
        Копия таблицы с новыми признаками "FamilySize" и "FamilyGroup"
    """

    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["FamilyGroup"] = pd.cut(
        df["FamilySize"],
        bins=[0, 1, 4, 20],
        labels=["Alone", "Small", "Large"]
    )
    
    return df