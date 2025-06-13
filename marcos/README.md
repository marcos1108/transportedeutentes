


# 1- treinado um modelo, k-means, para agrupar os dados
em clusters. O modelos realiza o agrupamento usando as variaveis temporais  do dataset.
features = [
        'MES', 'DIA_SEMANA',
        'HORA_SAIDA_NUM', 'MINUTOS_SAIDA', 'TURNO'
    ]
model_<num_clusters>.joblib=bv_k-means_1.py(input,num_clusters)

# 2- gerado um novo dataset em que a cada entrada 
juntou-se o numero do cluster. Os clusters foram
criados usando o k-means criado anteriormente
dataset_<num_clusters>.csv=bv_k-means_1.py(model_<num_clusters>.joblib)

# 3- Gerado o dataset final, em que foi criada a variavel 
'NUM_VEICULOS_CLUSTER'=soma('PED_NUM_VEICULOS',cluster)/num_eleme
dataset_final.csv=bv_k-means_3.py(dataset_<num_clusters>.csv,model_<num_clusters>.joblib)

# 4- criado (pipeline) e treinado classificador que faz predição da necessidade de viaturas

# 5- script preditor, perante a entrada de dados faz predição do numero de veiculoscom base na clasificação que fez da entrada 
