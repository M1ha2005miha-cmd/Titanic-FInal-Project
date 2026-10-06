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