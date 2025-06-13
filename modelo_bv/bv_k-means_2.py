import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


def main():
    # Caminhos fixos
    n_clusters = int(input("Introduza o número de clusters pretendido: "))
    input_csv = "./dataset.csv"
    model_path = f"./model_{n_clusters}.joblib"
    output_csv = f"./dataset_{n_clusters}.csv"

    # Carregar dataset
    df = pd.read_csv(input_csv)

    # Selecionar as mesmas variáveis utilizadas no modelo
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

    # Codificar 'TURNO' se for texto (mesmo procedimento do treino)
    #if X['TURNO'].dtype == 'object':
    #    le = LabelEncoder()
    #    X['TURNO'] = le.fit_transform(X['TURNO'])

    # Carregar pipeline/modelo treinado
    pipeline = joblib.load(model_path)

    # Prever clusters
    clusters = pipeline.predict(X)

    # Adicionar coluna 'cluster' no DataFrame original
    df['cluster'] = clusters

    # Guardar novo dataset
    df.to_csv(output_csv, index=False)
    print(f"Novo dataset guardado em '{output_csv}'")


if __name__ == "__main__":
    main()
