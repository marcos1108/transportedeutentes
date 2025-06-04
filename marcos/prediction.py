import joblib
import pandas as pd
import numpy as np

# Carregar os modelos salvos
rf_model = joblib.load('random_forest_model.pkl')
pca_model = joblib.load('pca_model.pkl')
scaler_model = joblib.load('scaler.pkl')

def prever_ambulancias(ano, mes, dia_semana, hora_saida_num, minutos_saida, turno):
    # Defina as colunas corretamente
    colunas_treino = ['ANO', 'MES', 'DIA_SEMANA', 'HORA_SAIDA_NUM', 'MINUTOS_SAIDA', 'TURNO']  # Ajuste conforme necessário

    # Criar o DataFrame com os dados fornecidos pelo utilizador
    dados_entrada = pd.DataFrame({
        'ANO': [ano],
        'MES': [mes],
        'DIA_SEMANA': [dia_semana],
        'HORA_SAIDA_NUM': [hora_saida_num],
        'MINUTOS_SAIDA': [minutos_saida],
        'TURNO': [turno]
    })

    # Garantir que a ordem das colunas seja a mesma que no treinamento
    dados_entrada = dados_entrada[colunas_treino]

    # Normalizar os dados de entrada com o scaler
    dados_entrada_normalizados = scaler_model.transform(dados_entrada)

    # Aplicar PCA para reduzir a dimensionalidade
    dados_entrada_pca = pca_model.transform(dados_entrada_normalizados)

    # Fazer a previsão com o modelo Random Forest
    previsao = rf_model.predict(dados_entrada_pca)

    # Exibir o resultado da previsão
    return previsao[0]

# Exemplo de como o utilizador pode introduzir os dados
ano = int(input("Introduza o ano: "))
mes = int(input("Introduza o mês: "))
dia_semana = int(input("Introduza o dia da semana (1=Segunda, 7=Domingo): "))
hora_saida_num = int(input("Introduza a hora de saída (em formato 24h): "))
minutos_saida = int(input("Introduza os minutos da hora de saída: "))
turno = int(input("Introduza o turno (1, 2, ou 3): "))

# Chamar a função para fazer a previsão
numero_ambulancias = prever_ambulancias(ano, mes, dia_semana, hora_saida_num, minutos_saida, turno)

# Exibir o resultado
print(f"A previsão do número de ambulâncias é: {numero_ambulancias}")
