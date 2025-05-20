# --- Treinar o modelo final de KMeans (k=6) com normalização e salvar arquivos necessários ---

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Carregar o dataset codificado
df = pd.read_csv("Dados/Dados_Preprocessados_Codificados.csv")

# 2. Normalizar os dados
scaler = StandardScaler()
df_normalizado = scaler.fit_transform(df)

# 3. Treinar o modelo com k=6
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(df_normalizado)

# 4. Salvar o modelo treinado e o normalizador
joblib.dump(kmeans, "modelo_kmeans.pkl")
joblib.dump(scaler, "normalizador.pkl")

# 5. Salvar os dados normalizados como CSV
df_normalizado_df = pd.DataFrame(df_normalizado, columns=df.columns)
df_normalizado_df.to_csv("Dados/dados_normalizados.csv", index=False)

print("Modelo, normalizador e dados normalizados salvos com sucesso.")
