# --- Gerar colunas de cluster de KMeans para análise exploratória usando dados normalizados ---

import pandas as pd
from sklearn.cluster import KMeans

# 1. Carregar os dados normalizados salvos anteriormente
df = pd.read_csv("Dados/dados_normalizados.csv")

# 2. Aplicar KMeans com diferentes valores de k (de 4 a 10)
for k in range(4, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df[f"cluster_{k}"] = kmeans.fit_predict(df)

# 3. Salvar o novo dataset com os clusters adicionados
df.to_csv("Dados/Dados_com_clusters.csv", index=False)

print("Clusters gerados e salvos em 'Dados_com_clusters.csv'")
