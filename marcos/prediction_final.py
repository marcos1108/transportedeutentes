import joblib
import pandas as pd
import numpy as np

# Carregar os modelos salvos
rf_model = joblib.load('random_forest_model.pkl')
pca_model = joblib.load('pca_model.pkl')
scaler_model = joblib.load('scaler.pkl')

# Carregar o dataset
rag = pd.read_csv('dataset.csv')

# Verifique as colunas do dataset
print("Colunas no dataset:", rag.columns)


# Função para fazer a previsão com base na entrada do utilizador
def prever_ambulancias(ano, mes, dia_semana, hora_saida_num, minutos_saida, turno):
    # Filtrar os dados de entrada (dados_entrada) com base nas variáveis fornecidas
    dados_filtrados = rag[(rag['MES'] == mes) &
                          (rag['DIA_SEMANA'] == dia_semana) &
                          (rag['HORA_SAIDA_NUM'] == hora_saida_num) &
                          (rag['MINUTOS_SAIDA'] == minutos_saida) &
                          (rag['TURNO'] == turno)]
                              
      
    # Calcular a mediana das variáveis relevantes para preencher os valores faltantes
    duracao_minutos_mediana = dados_filtrados['DURACAO_MINUTOS'].median() if not dados_filtrados['DURACAO_MINUTOS'].empty else 0
    ped_cod_servico_mediana = dados_filtrados['PED_COD_SERVICO'].median() if not dados_filtrados['PED_COD_SERVICO'].empty else 0
    ped_num_veiculos = 0
    ped_kilometros_mediana = dados_filtrados['PED_KILOMETROS'].median() if not dados_filtrados['PED_KILOMETROS'].empty else 0
    ped_local_mediana = dados_filtrados['PED_LOCAL'].median() if not dados_filtrados['PED_LOCAL'].empty else 0
    ped_num_bombeiros_mediana = dados_filtrados['PED_NUM_BOMBEIROS'].median() if not dados_filtrados['PED_NUM_BOMBEIROS'].empty else 0
    reserva_descricao_mediana = 0  # Preencher com valor padrão ou mediana
    reserva_destino_mediana = 0    # Preencher com valor padrão ou mediana
    reserva_posicao_mediana = 0    # Preencher com valor padrão ou mediana
    
    # Criar o DataFrame com os dados fornecidos pelo utilizador, ignorando o 'ANO' para os cálculos
    dados_entrada = pd.DataFrame({
        'ANO': [ano],  # Valor fornecido pelo utilizador, mas será mantido para a normalização
        'MES': [mes],  # Valor fornecido pelo utilizador
        'DIA_SEMANA': [dia_semana],  # Valor fornecido pelo utilizador
        'HORA_SAIDA_NUM': [hora_saida_num],  # Valor fornecido pelo utilizador
        'MINUTOS_SAIDA': [minutos_saida],  # Valor fornecido pelo utilizador
        'TURNO': [turno],  # Valor fornecido pelo utilizador
        'DURACAO_MINUTOS': [duracao_minutos_mediana],  # Mediana calculada
        'PED_COD_SERVICO': [ped_cod_servico_mediana],  # Mediana calculada
        'PED_NUM_VEICULOS': [ped_num_veiculos],  # Preenchido com valor padrão
        'PED_KILOMETROS': [ped_kilometros_mediana],  # Mediana calculada
        'PED_LOCAL': [ped_local_mediana],  # Mediana calculada
        'PED_NUM_BOMBEIROS': [ped_num_bombeiros_mediana],  # Mediana calculada
        'RESERVA_DESCRICAO': [reserva_descricao_mediana],  # Preenchido com valor padrão
        'RESERVA_DESTINO': [reserva_destino_mediana],  # Preenchido com valor padrão
        'RESERVA_POSICAO': [reserva_posicao_mediana]  # Preenchido com valor padrão
    })

    # Adicionar as colunas faltantes (que estão em rag, mas não em dados_entrada) com valores padrão
    colunas_faltantes = ['RESERVA_TER_COD', 'RESERVA_LOCAL', 'VEICULO', 'RESERVA_TIPO_FACT']
    for col in colunas_faltantes:
        dados_entrada[col] = 0  # Preencher com 0 ou NaN, conforme necessário
    
    # Garantir que a ordem das colunas seja a mesma que no treinamento
    colunas_treino = ['RESERVA_TER_COD', 'RESERVA_LOCAL', 'RESERVA_DESTINO', 'RESERVA_DESCRICAO', 'RESERVA_POSICAO',
                      'VEICULO', 'RESERVA_TIPO_FACT', 'PED_COD_SERVICO', 'PED_NUM_VEICULOS','PED_NUM_BOMBEIROS', 'PED_KILOMETROS', 
                      'PED_LOCAL', 'DURACAO_MINUTOS', 'ANO', 'MES', 'DIA_SEMANA', 'HORA_SAIDA_NUM', 'MINUTOS_SAIDA', 'TURNO']

    # Verifique se essas colunas estão presentes no seu DataFrame
    print("Colunas no DataFrame de entrada:", dados_entrada.columns, len(dados_entrada))
    print("Colunas no DataFrame de entrada:", colunas_treino,len(colunas_treino))
  
    # Ajuste a lista conforme necessário se algumas colunas estiverem faltando
    dados_entrada = dados_entrada[colunas_treino]
    print("DADOS ENTRADA")
    print("Colunas no DataFrame de entrada:", dados_entrada.columns, len(dados_entrada))
     
    # Normalizar os dados de entrada (mantendo a variável 'ANO')
    dados_entrada_normalizados = scaler_model.transform(dados_entrada)
    
    dados_entrada_norm = dados_entrada.drop(columns=['PED_NUM_VEICULOS'])  
    
    # Aplicar PCA para reduzir a dimensionalidade
    dados_entrada_pca = pca_model.transform(dados_entrada_norm)
     
    # Fazer a previsão com o modelo Random Forest
    previsao = rf_model.predict(dados_entrada_pca)
    
    # Exibir o resultado da previsão
    return previsao[0]
    """
    return "ola mundo"
    """
# Função principal
def main():
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

# Chamar a função principal
if __name__ == "__main__":
    main()
