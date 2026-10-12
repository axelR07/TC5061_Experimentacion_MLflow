# TC5061 - Experimentación con MLflow

## Actividad individual - Operaciones de Aprendizaje Automático

**Alumno:** Axel Fernando Rábago Díaz  
**Materia:** TC5061 - Operaciones de Aprendizaje Automático  
**Proyecto:** Experimentación y seguimiento de modelos con MLflow

## Descripción

Este proyecto corresponde a la actividad individual de experimentación con MLflow. El objetivo es migrar el flujo de entrenamiento desarrollado previamente en un Jupyter Notebook hacia una estructura modular y reproducible basada en Cookiecutter Data Science.

Se utiliza el dataset Heart Disease para entrenar y evaluar un modelo `RandomForestClassifier`. Los experimentos son registrados mediante MLflow Tracking, incluyendo hiperparámetros, métricas de evaluación, matriz de confusión y el modelo entrenado.

## Estructura del proyecto

```text
TC5061_Experimentacion_MLflow/
│
├── data/
│   ├── raw/
│   │   └── heart.csv
│   └── processed/
│       └── heart_clean.csv
│
├── notebooks/
│   └── TC5061_Notebook_Baseline_RabagoAxel.ipynb
│
├── reports/
│   └── figures/
│       └── confusion_matrix.png
│
├── TareaSemana4/
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   └── modeling/
│       └── train.py
│
├── requirements.txt
└── README.md
```

## Requisitos

El proyecto fue desarrollado y probado con:

- Python 3.14.7
- MLflow 3.17.0
- scikit-learn 1.9.1
- pandas 3.0.6
- NumPy 2.5.4
- Matplotlib 3.11.2

Las dependencias necesarias se encuentran documentadas en `requirements.txt`.

## Instalación

Crear un entorno virtual:

```powershell
python -m venv .venv
```

Activarlo en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install -r requirements.txt
```

## Preparación de los datos

El archivo original `heart.csv` debe encontrarse en:

```text
data/raw/heart.csv
```

Para ejecutar el proceso de limpieza:

```powershell
python -m TareaSemana4.dataset
```

El proceso genera:

```text
data/processed/heart_clean.csv
```

Los datos originales de `data/raw/` se mantienen sin modificaciones.

## Entrenamiento del modelo

Para ejecutar el modelo utilizando los hiperparámetros predeterminados:

```powershell
python -m TareaSemana4.modeling.train
```

Los valores predeterminados son:

```text
n_estimators = 100
max_depth = None
min_samples_split = 2
random_state = 42
```

También es posible modificar los hiperparámetros desde la línea de comandos. Ejemplo:

```powershell
python -m TareaSemana4.modeling.train --n_estimators 200 --random_state 42
```

Para consultar todas las opciones disponibles:

```powershell
python -m TareaSemana4.modeling.train --help
```

## Seguimiento de experimentos con MLflow

Para visualizar las corridas registradas:

```powershell
mlflow ui
```

Posteriormente, abrir en el navegador:

```text
http://127.0.0.1:5000
```

El experimento utilizado es:

```text
Heart_Disease_RandomForest
```

En cada corrida se registran:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `random_state`
- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión
- Modelo entrenado

## Reproducibilidad

Se utiliza `random_state=42` tanto para la división de los datos como para el modelo Random Forest. Esto permite mantener constantes las condiciones aleatorias entre las diferentes configuraciones experimentales y realizar una comparación consistente de los resultados.

Las versiones de las principales dependencias utilizadas durante la experimentación están fijadas en `requirements.txt`.