# Previsão de Necessidade de Ambulâncias - Bombeiros

Este projeto foi desenvolvido para prever a quantidade necessária de viaturas de transporte de doentes com base em dados históricos de transporte, utilizando **Machine Learning supervisionado com Random Forest**.

O projeto envolve:
- Pré-processamento de dados históricos dos bombeiros (2005 a 2024);
- Treinamento de um modelo Random Forest com redução de dimensionalidade (PCA);
- Desenvolvimento de uma interface interativa via Streamlit;
- Sistema de previsão operacional diária de transporte.

---

## 📂 Estrutura do Projeto

```
.
├── Dados/
│   ├── dataset.csv
│   ├── dataset_final.csv
├── best_rf_classifier_pipeline.pkl  (modelo final treinado)
├── prediction_funcional.py          (código de predição encapsulado como função)
├── app_streamlit_rf_v3.py           (aplicação final em Streamlit)
├── requirements.txt                 (dependências)
└── README.md
```

---

## 🚀 Como executar o projeto

### 1️⃣ Clonar o repositório

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

### 2️⃣ Instalar as dependências

Recomenda-se usar um ambiente virtual.

```bash
pip install -r requirements.txt
```

### 3️⃣ Executar a aplicação Streamlit

```bash
streamlit run app_streamlit_rf_v3.py
```

A aplicação abrirá no navegador, permitindo ao utilizador introduzir a data e hora da ocorrência e obter a previsão de viaturas necessárias.

---

## 🧠 Descrição técnica do modelo

- Algoritmo: Random Forest Classifier
- Redução de dimensionalidade: PCA (Análise de Componentes Principais)
- Pré-processamento: StandardScaler embutido no pipeline
- Entrada do modelo:
  - ANO, MES, DIA_SEMANA, HORA_SAIDA_NUM, MINUTOS_SAIDA, TURNO
  - Outras variáveis do histórico codificadas internamente (preenchidas como 0 na predição)

---

## 📊 Objetivo final

Auxiliar o corpo de bombeiros na previsão da quantidade de ambulâncias necessárias com base no perfil de demanda esperado, oferecendo uma ferramenta prática de apoio à tomada de decisão operacional.

