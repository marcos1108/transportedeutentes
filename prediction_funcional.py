import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# Carrega o modelo (já treinado, com scaler e PCA embutidos)
tmp = joblib.load('marcos/best_rf_classifier_pipeline.pkl')

if isinstance(tmp, Pipeline):
    model = tmp
    try:
        feature_names = model.named_steps['scaler'].feature_names_in_
    except Exception:
        feature_names = model.named_steps['scaler'].get_feature_names_out()
else:
    model = tmp
    feature_names = None

# Função principal agora encapsulada:
def prever_ambulancias(ano, mes, dia_semana, hora_saida, minutos_saida, turno):
    # Construir DataFrame inicial apenas com inputs recebidos
    data = pd.DataFrame([{  
        'ANO': ano,
        'MES': mes,
        'DIA_SEMANA': dia_semana,
        'HORA_SAIDA_NUM': hora_saida,
        'MINUTOS_SAIDA': minutos_saida,
        'TURNO': turno
    }])

    # Garante que todas as colunas esperadas estão presentes, preenchendo as faltantes com zero
    if feature_names is not None:
        for col in feature_names:
            if col not in data.columns:
                data[col] = 0
        data = data[feature_names]

    # Faz a previsão
    y_pred = model.predict(data)
    return y_pred[0]

# (Opcional) Interface antiga de linha de comando - pode ser removida caso use apenas Streamlit
if __name__ == "__main__":
    ano = int(input('Ano (ex: 2025): '))
    mes = int(input('Mês (1-12): '))
    dia_semana = int(input('Dia da semana (0=Segunda ... 6=Domingo): '))
    hora_saida = int(input('Hora de saída (0-23): '))
    minutos_saida = int(input('Minutos de saída (0-1439): '))
    turno = int(input('Turno (ex: Manhã (1), Tarde (2), Noite (3)): '))

    resultado = prever_ambulancias(ano, mes, dia_semana, hora_saida, minutos_saida, turno)
    print(f"No período temporal indicado,\nPrevê-se ser necessário : {resultado} veículos")
