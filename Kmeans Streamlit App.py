import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import datetime

# --- Carregar modelo e mapeamento ---
kmeans = joblib.load("modelo_kmeans.pkl")
with open("Dados/mapeamento.json", "r", encoding="utf-8") as f:
    mapeamentos_legiveis = json.load(f)

st.title("Classificação de Transporte - Clustering K-Means")
st.write("Preencha os dados do transporte para classificar em um cluster.")

# --- Inputs categóricos ---
reserva_ter_cod = st.selectbox("Terminal (RESERVA_TER_COD)", options=mapeamentos_legiveis["RESERVA_TER_COD"])
reserva_local = st.selectbox("Local de origem (RESERVA_LOCAL)", options=mapeamentos_legiveis["RESERVA_LOCAL"])
reserva_destino = st.selectbox("Destino (RESERVA_DESTINO)", options=mapeamentos_legiveis["RESERVA_DESTINO"])
reserva_descricao = st.selectbox("Descrição (RESERVA_DESCRICAO)", options=mapeamentos_legiveis["RESERVA_DESCRICAO"])
reserva_posicao = st.selectbox("Posição do utente (RESERVA_POSICAO)", options=mapeamentos_legiveis["RESERVA_POSICAO"])
veiculo = st.selectbox("Veículo utilizado", options=mapeamentos_legiveis["VEICULO"])
reserva_tipo_fact = st.selectbox("Tipo de faturação", options=mapeamentos_legiveis["RESERVA_TIPO_FACT"])
ped_cod_servico = st.number_input("Código do serviço (PED_COD_SERVICO)", min_value=1, step=1)
ped_num_veiculos = st.number_input("Nº de viaturas (PED_NUM_VEICULOS)", min_value=1, step=1)
ped_num_bombeiros = st.number_input("Nº de bombeiros (PED_NUM_BOMBEIROS)", min_value=1, step=1)
ped_kilometros = st.number_input("Distância (PED_KILOMETROS)", min_value=0.0, step=0.1)
ped_local = st.selectbox("Local do pedido (PED_LOCAL)", options=mapeamentos_legiveis["PED_LOCAL"])

# --- Inputs temporais ---
duracao_minutos = st.number_input("Duração (minutos)", min_value=0)
ano = st.number_input("Ano", min_value=2005, max_value=2024, step=1)
mes = st.number_input("Mês", min_value=1, max_value=12, step=1)
dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
dia_semana_legivel = st.selectbox("Dia da semana", options=dias_semana)
dia_semana = dias_semana.index(dia_semana_legivel)

hora_completa = st.time_input("Hora de saída", value=datetime.time(8, 0))
hora_saida_num = hora_completa.hour + hora_completa.minute / 60
minutos_saida = hora_completa.hour * 60 + hora_completa.minute

# --- Cálculo automático do turno ---
if 6 <= hora_completa.hour < 12:
    turno_nome = "Manha"
elif 12 <= hora_completa.hour < 18:
    turno_nome = "Tarde"
elif 18 <= hora_completa.hour < 24:
    turno_nome = "Noite"
else:
    turno_nome = "Madrugada"

# --- Botão de classificação ---
if st.button("Classificar Transporte"):
    entrada = pd.DataFrame([[
        mapeamentos_legiveis["RESERVA_TER_COD"].index(reserva_ter_cod) + 1,
        mapeamentos_legiveis["RESERVA_LOCAL"].index(reserva_local) + 1,
        mapeamentos_legiveis["RESERVA_DESTINO"].index(reserva_destino) + 1,
        mapeamentos_legiveis["RESERVA_DESCRICAO"].index(reserva_descricao) + 1,
        mapeamentos_legiveis["RESERVA_POSICAO"].index(reserva_posicao) + 1,
        mapeamentos_legiveis["VEICULO"].index(veiculo) + 1,
        mapeamentos_legiveis["RESERVA_TIPO_FACT"].index(reserva_tipo_fact) + 1,
        ped_cod_servico, ped_num_veiculos, ped_num_bombeiros, ped_kilometros,
        mapeamentos_legiveis["PED_LOCAL"].index(ped_local) + 1,
        duracao_minutos, ano, mes, dia_semana,
        hora_saida_num, minutos_saida,
        mapeamentos_legiveis["TURNO"].index(turno_nome) + 1
    ]], columns=[
        "RESERVA_TER_COD", "RESERVA_LOCAL", "RESERVA_DESTINO", "RESERVA_DESCRICAO",
        "RESERVA_POSICAO", "VEICULO", "RESERVA_TIPO_FACT", "PED_COD_SERVICO",
        "PED_NUM_VEICULOS", "PED_NUM_BOMBEIROS", "PED_KILOMETROS", "PED_LOCAL",
        "DURACAO_MINUTOS", "ANO", "MES", "DIA_SEMANA",
        "HORA_SAIDA_NUM", "MINUTOS_SAIDA", "TURNO"
    ])

    cluster = kmeans.predict(entrada)[0]
    st.success(f"Este transporte pertence ao cluster {cluster}.")
