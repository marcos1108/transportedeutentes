# --- Calcular veículos necessários por ano com base na média anual e picos diários ---

import pandas as pd

# 1. Carregar os dados por dia
df = pd.read_csv("Dados/Dados_Com_Variaveis_Derivadas.csv")

# 2. Calcular o valor máximo de DIFERENCA_DIARIA_VEICULOS por ano
maximos = df.groupby("ANO")["DIFERENCA_DIARIA_VEICULOS"].max()

# 3. Calcular os veículos necessários por ano
df_resultado = df.groupby("ANO")[["MEDIA_ANUAL_VEICULOS"]].first().copy()
df_resultado["MAXIMO_DIFERENCA_DIARIA"] = maximos
df_resultado["VEICULOS_NECESSARIOS_ANO"] = df_resultado["MEDIA_ANUAL_VEICULOS"] + df_resultado["MAXIMO_DIFERENCA_DIARIA"]

# 4. Arredondar resultado final
df_resultado["VEICULOS_NECESSARIOS_ANO"] = df_resultado["VEICULOS_NECESSARIOS_ANO"].round(2)

# 5. Salvar como CSV
df_resultado.to_csv("veiculos_necessarios_ano.csv")

# 6. Imprimir resultado
print("\nVeículos necessários por ano (estimativa):\n")
print(df_resultado)
