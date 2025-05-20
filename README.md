# K-Means Streamlit App

Este projeto é uma aplicação web desenvolvida com Streamlit para identificar o cluster ao qual pertence um transporte de utente, com base em um modelo K-Means treinado.

## Estrutura

- `Kmeans Streamlit App.py`: aplicação web interativa
- `modelo_kmeans.pkl`: modelo KMeans treinado com k=6
- `normalizador.pkl`: scaler usado para normalização
- `mapeamento_legivel.json`: mapeamento de valores legíveis para códigos

## Como usar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute o app:
```bash
streamlit run Kmeans Streamlit App.py
```

3. Preencha os dados e descubra a qual cluster o transporte pertence!

## Requisitos

- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- joblib
