from pathlib import Path

import pandas as pd
from loguru import logger

from TareaSemana4.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def load_data(input_path: Path) -> pd.DataFrame:
    """Carga el dataset original de Heart Disease."""
    logger.info(f"Cargando datos desde: {input_path}")
    return pd.read_csv(input_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia los valores inválidos identificados durante el EDA."""
    df = df.copy()

    # Calculamos la mediana del colesterol ignorando valores iguales a 0.
    median_cholesterol = df.loc[
        df["Cholesterol"] > 0, "Cholesterol"
    ].median()

    # Eliminamos el registro con presión arterial en reposo igual a 0.
    df = df[df["RestingBP"] != 0].copy()

    # Sustituimos colesterol igual a 0 por la mediana calculada.
    df.loc[df["Cholesterol"] == 0, "Cholesterol"] = median_cholesterol

    logger.info(
        f"Datos limpios: {df.shape[0]} filas y {df.shape[1]} columnas."
    )

    return df


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda el dataset limpio en la carpeta de datos procesados."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.success(f"Dataset procesado guardado en: {output_path}")


def main() -> None:
    """Ejecuta el proceso de carga, limpieza y almacenamiento."""
    input_path = RAW_DATA_DIR / "heart.csv"
    output_path = PROCESSED_DATA_DIR / "heart_clean.csv"

    df = load_data(input_path)
    df_clean = clean_data(df)
    save_data(df_clean, output_path)


if __name__ == "__main__":
    main()