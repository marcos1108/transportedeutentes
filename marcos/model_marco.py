# Importação de bibliotecas necessáriasimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# Carregar o dataset
df = pd.read_csv('dataset.csv')

# 1. Pré-processamento
# Verificar valores nulos
print(df.isnull().sum())  # Exibe o número de valores nulos em cada coluna
df = df.dropna()  # Remover linhas com valores nulos, pode substituir por uma técnica de imputação se necessário

# Selecionar as colunas numéricas e categóricas
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# 2. Codificação de variáveis categóricas (se houver)
# Exemplo para uma variável categórica hipotética, se necessário
# df = pd.get_dummies(df, columns=['RESERVA_DESCRICAO', 'PED_LOCAL'])

# 3. Normalização dos dados
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 4. Separar as features (X) e o alvo (y)
X = df.drop(columns=['PED_NUM_VEICULOS'])  # Remover a variável alvo
y = df['PED_NUM_VEICULOS']  # Variável alvo

# 5. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Aplicar PCA para redução de dimensionalidade
pca = PCA(n_components=5)  # Vamos reduzir para 5 componentes principais (ajuste conforme necessário)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 7. Treinamento do modelo Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)

# 8. Previsão e avaliação do modelo
y_pred = rf.predict(X_test_pca)

# Avaliação do desempenho
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Calcular o RMSE manualmente
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# 9. Importância das variáveis no modelo Random Forest
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Importância das Variáveis")
plt.bar(range(X_train_pca.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_pca.shape[1]), indices)
plt.xlabel("Componente Principal")
plt.ylabel("Importância")
plt.show()

# 10. Salvamento do modelo e do PCA
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(pca, 'pca_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Modelos salvos com sucesso!")
