import pandas as pd
import os

# Solicita ao usuário o número de clusters desejado
n_clusters = int(input("Introduza o número de clusters pretendido: "))

# Monta o nome do arquivo de entrada e verifica se existe
input_path = f"./dataset_{n_clusters}.csv"
if not os.path.isfile(input_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

# Lê o dataset
df = pd.read_csv(input_path)

# 1. Conta o total de entradas de cada cluster
entries_per_cluster = df['cluster'].value_counts()
min_value_clusters = entries_per_cluster.min()
print("Contagem de entradas por cluster:\n", entries_per_cluster)
print(f"Menor número de entradas entre os clusters: {min_value_clusters}")

# 2. Soma os valores da variável PED_NUM_VEICULOS por cluster
#    e prepara uma série para mapear diretamente
veiculos_por_cluster = (
    df
    .groupby('cluster')['PED_NUM_VEICULOS']
    .sum()
)
print("Total de veículos por cluster:\n", veiculos_por_cluster)

# 3. Cria a variável NUM_VEICULOS_CLUSTER
#    Cada célula recebe (soma de veículos do seu cluster) / min_value_clusters
#    Arredonda o resultado e converte para inteiro
ratio = df['cluster'].map(veiculos_por_cluster) / min_value_clusters
df['NUM_VEICULOS_CLUSTER'] = ratio.round().astype(int)
print(df['NUM_VEICULOS_CLUSTER'])
# 4. Salva o resultado em um novo arquivo
output_path = 'dataset_final.csv'
df.to_csv(output_path, index=False)
print(f"Arquivo '{output_path}' gerado com sucesso.")
