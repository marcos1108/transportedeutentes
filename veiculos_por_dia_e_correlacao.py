# --- Criar variáveis derivadas por dia e gerar correlação com o dataset completo ---

import pandas as pd

# 1. Carregar os dados com datas e os dados codificados
df_codificado = pd.read_csv("Dados/Dados_Preprocessados_Codificados.csv")
df_com_datas = pd.read_csv("Dados/Dados_Preprocessados.csv")

# 2. Garantir que a data esteja presente no codificado
df_codificado["PED_DATA"] = pd.to_datetime(df_com_datas["PED_DATA"], errors='coerce')

# 3. Somar PED_NUM_VEICULOS por dia
veiculos_por_dia = df_codificado.groupby("PED_DATA")["PED_NUM_VEICULOS"].sum().reset_index()
veiculos_por_dia.rename(columns={"PED_NUM_VEICULOS": "TOTAL_VEICULOS_DIA"}, inplace=True)
veiculos_por_dia["ANO"] = veiculos_por_dia["PED_DATA"].dt.year

# 4. Calcular a média anual de veículos por dia
media_anual = veiculos_por_dia.groupby("ANO")["TOTAL_VEICULOS_DIA"].mean().to_dict()
veiculos_por_dia["MEDIA_ANUAL_VEICULOS"] = veiculos_por_dia["ANO"].map(media_anual)

# 5. Calcular a diferença diária
veiculos_por_dia["DIFERENCA_DIARIA_VEICULOS"] = (
    veiculos_por_dia["TOTAL_VEICULOS_DIA"] - veiculos_por_dia["MEDIA_ANUAL_VEICULOS"]
)

# 6. Mesclar com o dataset original (por PED_DATA)
df_final = pd.merge(df_codificado, veiculos_por_dia[["PED_DATA", "MEDIA_ANUAL_VEICULOS", "DIFERENCA_DIARIA_VEICULOS"]],
                    on="PED_DATA", how="left")

# 7. Salvar o dataset completo com as novas colunas
df_final.to_csv("Dados/Dados_Com_Variaveis_Derivadas.csv", index=False)

# 8. Gerar a matriz de correlação numérica
correlacao = df_final.drop(columns=["PED_DATA"]).corr(numeric_only=True)
correlacao.to_csv("Dados/correlacoes_derivadas_completas.csv")

print("Colunas derivadas adicionadas e correlações salvas com sucesso.")

# 9. Mostrar no terminal as variáveis mais correlacionadas para cada coluna
print("\n=== Maiores correlações entre variáveis (ignorando autovariáveis) ===")
limite = 0.6

for var in correlacao.columns:
    correladas = correlacao[var].drop(labels=[var]).abs()
    mais_fortes = correladas[correladas > limite].sort_values(ascending=False)
    if not mais_fortes.empty:
        print(f"\n{var} se correlaciona fortemente com:")
        for outra_var in mais_fortes.index:
            valor_real = correlacao.at[outra_var, var]
            print(f"  - {outra_var}: {valor_real:.2f}")


