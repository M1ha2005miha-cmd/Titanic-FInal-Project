import yaml
from sklearn.metrics import accuracy_score

from src.data import load_data
from src.features import build_features
from src.preprocessing import build_preprocessor
from src.validation import build_cv, compare_models, tune_model
from src.submission import make_submission
from src.models import build_voting, predict_by_sex, get_models, build_pipeline, build_stacking


def main():
    """Сравнивает модели кросс-валидацией и сохраняет их в таблицу результатов."""

    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    
    train_data, test_data = load_data(
        config["data"]["training_path"],
        config["data"]["test_path"]
    )

    X = build_features(train_data).drop(columns="Survived")
    y = train_data["Survived"]

    preprocessor = build_preprocessor(
        config["features"]["numeric"],
        config["features"]["categorical"]
    )
   
    cv = build_cv(
        config["validation"]["n_splits"],
        config["validation"]["random_state"]
    )

    models = get_models(config["validation"]["random_state"])
    for name, param_grid in config["tuning"].items():
        pipeline = build_pipeline(preprocessor, models[name])
        search = tune_model(pipeline, param_grid, X, y, cv)
        print(f"{name}:{search.best_params_}, accuracy = {search.best_score_:.3f}")
        models[f"{name}_tuned"] = search.best_estimator_.named_steps["model"]

    models["voting"] = build_voting(models)
    models["stacking"] = build_stacking(models)
    
    baseline = accuracy_score(y, predict_by_sex(train_data))
    print(f"Baseline(правило по полу): accuracy = {baseline:.3f}")

    results = compare_models(
        models, preprocessor, X, y, cv, config["validation"]["metrics"]
    )
    print(results.round(3))
    results.to_csv(config["output"]["results_path"])

    X_test = build_features(test_data)

    for name in config["submission"]["models"]:
        path = f"{config["submission"]["output_dir"]}/submission_{name}.csv"
        make_submission(models[name], preprocessor, X, y, X_test, test_data["PassengerId"], path)
        print(f"Сохранен {path}")
    
   


if __name__ == "__main__":
    main()

