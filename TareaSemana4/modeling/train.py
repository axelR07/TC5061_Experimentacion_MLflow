import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from TareaSemana4.config import PROCESSED_DATA_DIR, REPORTS_DIR
from TareaSemana4.features import build_preprocessor


EXPERIMENT_NAME = "Heart_Disease_RandomForest"


def parse_args():
    """Recibe los hiperparámetros del modelo desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de Random Forest para Heart Disease."
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Número de árboles del Random Forest.",
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Profundidad máxima de los árboles.",
    )

    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="Número mínimo de muestras requeridas para dividir un nodo.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Semilla aleatoria para garantizar reproducibilidad.",
    )

    return parser.parse_args()


def load_processed_data(data_path: Path):
    """Carga el dataset limpio y separa variables predictoras y objetivo."""
    df = pd.read_csv(data_path)

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    return X, y


def build_model(args):
    """Construye el pipeline de preprocesamiento y clasificación."""
    preprocessor = build_preprocessor()

    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo y devuelve sus principales métricas."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    return metrics, y_pred


def save_confusion_matrix(y_test, y_pred):
    """Genera y guarda la matriz de confusión como artefacto."""
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_path = figures_dir / "confusion_matrix.png"

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
    )

    display.ax_.set_title("Matriz de confusión - Random Forest")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()

    return figure_path


def main():
    args = parse_args()

    data_path = PROCESSED_DATA_DIR / "heart_clean.csv"

    X, y = load_processed_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_model(args)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("random_state", args.random_state)

        model.fit(X_train, y_train)

        metrics, y_pred = evaluate_model(
            model,
            X_test,
            y_test,
        )

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        confusion_matrix_path = save_confusion_matrix(
            y_test,
            y_pred,
        )

        mlflow.log_artifact(
            str(confusion_matrix_path),
            artifact_path="figures",
        )

        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print("\nResultados de la corrida")
        print("-------------------------")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")
        print("-------------------------")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()