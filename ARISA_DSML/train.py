from pathlib import Path  # Dodajemy import Path

import joblib  # Dodajemy import joblib
import optuna
import plotly.graph_objects as go
from catboost import CatBoostClassifier, Pool, cv
from preproc import download_data, preprocess_data
from sklearn.model_selection import train_test_split


def train_model(X_train, y_train, categorical_indices):
    """
    Funkcja do trenowania modelu CatBoost z wykorzystaniem optymalizacji hiperparametrów.
    """  # noqa: F821
    best_params_path = "results/best_params.pkl"

    if not Path(best_params_path).is_file():
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42
        )

        def objective(trial):
            params = {
                "depth": trial.suggest_int("depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3),
                "iterations": trial.suggest_int("iterations", 100, 300),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 100.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.01, 1),
                "random_strength": trial.suggest_float("random_strength", 1e-5, 100.0, log=True),
            }
            model = CatBoostClassifier(**params, verbose=0)
            model.fit(
                X_train_opt,
                y_train_opt,
                eval_set=(X_val_opt, y_val_opt),
                cat_features=categorical_indices,
                early_stopping_rounds=50,
            )
            return model.get_best_score()["validation"]["Logloss"]

        study = optuna.create_study(direction="minimize")  # noqa: F821
        study.optimize(objective, n_trials=50)

        joblib.dump(study.best_params, best_params_path)  # noqa: F821
        params = study.best_params
    else:
        params = joblib.load(best_params_path)

    print("Best Parameters:", params)

    params["eval_metric"] = "F1"
    params["loss_function"] = "Logloss"

    model = CatBoostClassifier(**params, verbose=True)
    data = Pool(X_train, y_train, cat_features=categorical_indices)

    cv_results = cv(
        params=params,
        pool=data,
        fold_count=5,
        partition_random_seed=42,
        shuffle=True,
    )

    cv_results.to_csv("results/cv_results.csv", index=False)

    # Wizualizacja wyników
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cv_results["iterations"],
            y=cv_results["test-F1-mean"],
            mode="lines",
            name="Mean F1 Score",
            line=dict(color="blue"),
        )
    )
    fig.write_image("results/f1_score.png")

    return model


def load_and_preprocess():
    """
    Funkcja do załadowania danych i przygotowania ich do treningu.
    """
    download_data()

    df_train = preprocess_data("data/titanic/train.csv")
    y_train = df_train.pop("Survived")
    X_train = df_train

    categorical = ["Pclass", "Sex", "Embarked", "Deck", "Title"]
    categorical_indices = [
        X_train.columns.get_loc(col) for col in categorical if col in X_train.columns
    ]

    return X_train, y_train, categorical_indices


def main():
    X_train, y_train, categorical_indices = load_and_preprocess()
    model = train_model(X_train, y_train, categorical_indices)
    model.save_model("results/final_model.cbm")


if __name__ == "__main__":
    main()
