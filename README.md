# Projeto de Análise de Transportes com K-Means

Este projeto utiliza técnicas de aprendizagem não supervisionada (K-Means) e engenharia de variáveis para analisar padrões em dados de transporte de utentes. A aplicação final permite prever a qual cluster um novo transporte pertence.

---

## 📁 Estrutura

- `Dados_Preprocessados_Codificados.csv`: dados originais codificados
- `Dados_Com_Variaveis_Derivadas.csv`: dados com novas variáveis por dia
- `modelo_kmeans.pkl`: modelo K-Means final com k=6
- `normalizador.pkl`: scaler usado na normalização
- `mapeamento_legivel.json`: tradução dos valores numéricos
- `Kmeans Streamlit App.py`: aplicação web interativa
- `treinar_modelo_final.py`: script que treina o modelo e normaliza os dados
- `gerar_clusters.py`: gera clusters exploratórios de k=4 a k=10
- `veiculos_por_dia_e_correlacao.py`: cria variáveis por dia e gera matriz de correlação
- `correlacoes_derivadas_completas.csv`: matriz de correlação entre todas as variáveis

---

## 🧪 Como executar

### Instale as dependências:
```bash
pip install -r requirements.txt
```

### Para treinar o modelo:
```bash
python treinar_modelo_final.py
```

### Para gerar clusters exploratórios:
```bash
python gerar_clusters.py
```

### Para gerar variáveis derivadas e correlação:
```bash
python veiculos_por_dia_e_correlacao.py
```

### Para rodar a aplicação web:
```bash
streamlit run Kmeans Streamlit App.py
```

---

## 👤 Autor
Marcos Ramos  
Projeto acadêmico no âmbito do curso de Análise de Dados – IPG
