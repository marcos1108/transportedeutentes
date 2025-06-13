


# 1- uso de modelo k-means para agrupar dados
treinado um modelo, k-means, para agrupar os dados em clusters. O modelos realiza o agrupamento usando 
as variaveis temporais  do dataset.

*features = [
        'MES', 'DIA_SEMANA',
        'HORA_SAIDA_NUM', 'MINUTOS_SAIDA', 'TURNO'
    ]*
	
*model_<num_clusters>.joblib=bv_k-means_1.py(input,num_clusters)*

# 2- geração de novo dataset 
gerado um novo dataset com uma nova coluna, ckuster. A cada entrada juntou-se o numero do cluster. 
Os clusters foram criados usando o k-means criado anteriormente


*dataset_<num_clusters>.csv=bv_k-means_1.py(model_<num_clusters>.joblib)*

# 3- Gerado o dataset final, em que foi criada a variavel 

'NUM_VEICULOS_CLUSTER'=soma('PED_NUM_VEICULOS',cluster)/min{num_elementos_cluster_j: j=1,...,num_cluster}

*dataset_final.csv=bv_k-means_3.py(dataset_<num_clusters>.csv,model_<num_clusters>.joblib)*

# 4- modelo

criado modelo classiificador (pipeline) e treinado (método fit) que faz predição da necessidade de viaturas.

O modelo criado é um modelos ajustado dataset entregue, outro dataset outro contexto produzir o mesmo modelos
metodologico mas com comportamento de predição diferente 

Modelo=
*best_rf_classifier_pipeline.pkl*

# 5- script preditor 

script preditor que solicita a entrada de dados temporais e prevê o numero de veiculos necessários para esse


