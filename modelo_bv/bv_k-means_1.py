import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib


def main():
    # Caminho fixo para o dataset
    input_csv = "./dataset.csv"

    # Solicitar interativamente o número de clusters ao utilizador
    while True:
        try:
            n_clusters = int(input("Introduza o número de clusters pretendido: "))
            if n_clusters > 0:
                break
            else:
                print("Por favor, introduza um número inteiro positivo.")
        except ValueError:
            print("Valor inválido. Introduza um número inteiro.")

    # Carregar dataset
    df = pd.read_csv(input_csv)

    # Selecionar variáveis temporais para clustering
    features = [
        'MES', 'DIA_SEMANA',
        'HORA_SAIDA_NUM', 'MINUTOS_SAIDA', 'TURNO'
    ]
    """
    features = [
        'ANO', 'MES', 'DIA_SEMANA',
        'HORA_SAIDA_NUM', 'MINUTOS_SAIDA', 'TURNO'
    ]
    """
    X = df[features].copy()

    # Codificar 'TURNO' se for texto
    # if X['TURNO'].dtype == 'object':
    #    le = LabelEncoder()
    #    X['TURNO'] = le.fit_transform(X['TURNO'])

    # Pipeline: escalamento + KMeans
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42))
    ])

    # Ajustar modelo
    pipeline.fit(X)

    # Gerar nome de ficheiro com data
    date_str = datetime.now().strftime("%Y%m%d")
    model_filename = f"model_{n_clusters}.joblib"

    # Guardar modelo
    joblib.dump(pipeline, model_filename)
    print(f"Modelo guardado em '{model_filename}'")


if __name__ == "__main__":
    main()
