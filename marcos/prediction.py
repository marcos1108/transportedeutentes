import pandas as pd
import joblib

# Carregar o pipeline treinado
tmp = joblib.load('./best_rf_classifier_pipeline.pkl')
# Se for Pipeline, extrai o modelo e os nomes de features esperados
from sklearn.pipeline import Pipeline
if isinstance(tmp, Pipeline):
    model = tmp
    # Escaler ou transformer pré-processamento deve expor feature_names_in_
    try:
        feature_names = model.named_steps['scaler'].feature_names_in_
    except Exception:
        # Tentativa alternativa: PCA não altera nomes; usa feature_names_in_ de first transformer
        feature_names = model.named_steps['scaler'].get_feature_names_out()
else:
    model = tmp
    feature_names = None

# Função para obter input do utilizador
def obter_input_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Por favor, insere um número inteiro válido.")

# Solicitar ao utilizador os valores das variáveis
print("Introduz os valores para prever o NUM_VEICULOS_CLUSTER:")
ano = obter_input_int('Ano (ex: 2025): ')
mes = obter_input_int('Mês (1-12): ')
dia_semana = obter_input_int('Dia da semana (0=Segunda ... 6=Domingo): ')
hora_saida = obter_input_int('Hora de saída (0-23): ')
minutos_saida = obter_input_int('Minutos de saída (0-1439): ') 
turno = input('Turno (ex: Manhã (1), Tarde (2), Noite (3)): ') # falta verificar o turno com os numeros

# Construir DataFrame inicial apenas com inputs brutos
data = pd.DataFrame([{  
    'ANO': ano,
    'MES': mes,
    'DIA_SEMANA': dia_semana,
    'HORA_SAIDA_NUM': hora_saida,
    'MINUTOS_SAIDA': minutos_saida,
    'TURNO': turno
}])

# Se o pipeline espera colunas dummy de 'TURNO', precisamos de criar corretamente
# Assumindo que o pipeline tem um OneHotEncoder/ColumnTransformer, basta passar data

# Caso contrário, reindexa para as features esperadas, preenchendo faltantes com zero
if feature_names is not None:
    # Garante que todas as colunas existem
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0
    # Reordena
    data = data[feature_names]

# Prever
y_pred = model.predict(data)
print(f"No período temporal indicado,\nPrevê-se ser necessário : {y_pred[0]} veículos")

