import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import joblib
from sklearn.model_selection import RandomizedSearchCV

# 1. Carregar e limpar
df = pd.read_csv('dataset_final.csv').dropna()
df['NUM_VEICULOS_CLUSTER'] = df['NUM_VEICULOS_CLUSTER'].astype(int)

# 2. Separar X e y
X = df.drop(columns=['NUM_VEICULOS_CLUSTER'])
y = df['NUM_VEICULOS_CLUSTER']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Pipeline de pré-processamento + classificador
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),                   # vamos afinar n_components
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# 5. Espaço de hiper-parâmetros
param_grid = {
    # PCA: sem PCA (None) ou vários números de componentes
    'pca__n_components': [None, 5, 10, 15],
    # Random Forest:
    'clf__n_estimators': [100, 200, 500],
    'clf__max_depth': [None, 5, 10, 20],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    # Experimenta também balanceamento de classes
    'clf__class_weight': [None, 'balanced']
}

# 6. GridSearchCV com scoring por F1 macro
grid = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_grid,
    n_iter=100,           # testa só 100 combinações
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2,
    random_state=42
)
grid.fit(X_train, y_train)

print("→ Melhores parâmetros:", grid.best_params_)
print("→ Melhor F1_macro em CV:", grid.best_score_)

# 7. Avaliação no conjunto de teste
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\n=== Métricas no Teste ===")
print(f"Acurácia:       {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall (macro):    {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1 (macro):        {f1_score(y_test, y_pred, average='macro'):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# 8. Guardar modelo completo
joblib.dump(best_model, 'best_rf_classifier_pipeline.pkl')
print("\nModelo guardado em 'best_rf_classifier_pipeline.pkl'")
