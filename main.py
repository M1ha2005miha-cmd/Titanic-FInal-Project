import yaml

from src.data import load_data


def main():
    """Загружает данные по путям из конфига и выводит их размеры."""

    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    
    train_path = config["data"]["training_path"]
    test_path = config["data"]["test_path"]
    
    train_data, test_data = load_data(train_path, test_path)
    print("train:", train_data.shape)
    print("test:", test_data.shape)


if __name__ == "__main__":
    main()

