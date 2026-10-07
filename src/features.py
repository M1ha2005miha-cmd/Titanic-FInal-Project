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


def add_has_cabin(df):
    """Добавляет новый признак "HasCabin" - известен ли номер каюты (1/0)

    Args:
        df: Таблица с данными пассажиров, в которой есть "Cabin"

    Returns:
        Копию таблицы с новым признаком "HasCabin"
    """

    df = df.copy()
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    
    return df


def build_features(df):
    """Добавляет все новые признаки: Title, FamilySize, FamilyGroup, HasCabiin.

    Args:
        df: таблица с данными пассажиров, в которой есть столбцы
            "Name", "SibSp", "Parch" и "Cabin".
    
    Returns:
        Копию таблицы с четырьмя новыми признаками.
    """

    df = add_title(df)
    df = add_family_features(df)
    df = add_has_cabin(df)

    return df