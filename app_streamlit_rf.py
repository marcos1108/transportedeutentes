import streamlit as st
from datetime import datetime
from prediction_funcional import prever_ambulancias

# Título do app
st.title("Previsão de Necessidade de Ambulâncias")

# Inputs de data e hora
st.header("Introduza a data e hora da ocorrência")
data_hora = st.date_input("Data prevista", value=datetime(2025, 1, 1))
hora = st.time_input("Hora prevista", value=datetime(2025, 1, 1, 0, 0).time())

# Processamento das variáveis derivadas
ano = data_hora.year
mes = data_hora.month
dia_semana = data_hora.weekday()
hora_saida = hora.hour
minutos_saida = hora.hour * 60 + hora.minute

# Regras simples para o turno
if 6 <= hora_saida < 14:
    turno = 1
elif 14 <= hora_saida < 22:
    turno = 2
else:
    turno = 3

# (opcional) Exibir debug dos dados derivados
st.write(f"Ano: {ano}, Mês: {mes}, Dia da Semana: {dia_semana}, Hora: {hora_saida}, Minutos: {minutos_saida}, Turno: {turno}")

# Botão de previsão
if st.button("Prever necessidade de viaturas"):
    previsao = prever_ambulancias(ano, mes, dia_semana, hora_saida, minutos_saida, turno)
    st.success(f"Prevê-se ser necessário: {previsao} veículos.")
