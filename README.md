# 🏥 Predição de Sepse com Machine Learning
## Projeto de Mineração de Dados - Metodologia CRISP-DM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PhysioNet](https://img.shields.io/badge/Dataset-PhysioNet%202019-red.svg)](https://physionet.org/content/challenge-2019/)

Este projeto implementa um pipeline completo de Machine Learning seguindo a metodologia CRISP-DM para predição precoce de sepse em pacientes de UTI, utilizando o dataset do PhysioNet 2019 Challenge.


---

## 📋 Índice

- [Dataset](#-dataset)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Metodologia CRISP-DM](#-metodologia-crisp-dm)
- [Modelos Implementados](#-modelos-implementados)
- [Pré-processamento](#-pré-processamento)
- [Resultados](#-resultados)
- [Instalação e Uso](#-instalação-e-uso)
- [Dicionário de Variáveis](#-dicionário-de-variáveis)

---

## 📊 Dataset

O dataset utilizado é do **PhysioNet 2019 Challenge**, contendo dados de pacientes internados em UTI com múltiplas variáveis clínicas coletadas ao longo do tempo.

### Características Principais

- **Total de registros**: 1.552.210 observações (40.336 pacientes)
- **Features**: 40 variáveis clínicas (sinais vitais, exames laboratoriais, dados demográficos)
- **Target**: Predição binária de sepse (SepsisLabel: 0=Não-Sepsis, 1=Sepsis)
- **Desbalanceamento**: ~98.2% Não-Sepsis vs 1.8% Sepsis
- **Dados temporais**: Múltiplas medições por paciente ao longo da internação (Hour 0-335+)
- **Missing values**: Presença significativa de valores ausentes em variáveis clínicas
- **Divisão**: 80% treino (31.475 pacientes) / 20% teste (8.861 pacientes)

### Critério Sepsis-3

A classificação de sepse segue a definição **Sepsis-3**, que considera disfunção orgânica potencialmente fatal causada por resposta desregulada do hospedeiro à infecção.

---

## 🗂️ Estrutura do Projeto

```
data-mining-proj-crisp-dm/
│
├── 📁 modeling/                          # Notebooks e scripts de modelagem
│   ├── 1-KNN_model_eval.ipynb           # K-Nearest Neighbors
│   ├── 2-LVQ_model_eval.ipynb           # Learning Vector Quantization
│   ├── 3-DecisionTree_model_eval.ipynb  # Árvore de Decisão
│   ├── 4-RandomForest_model_eval.ipynb  # Random Forest
│   ├── 5-SVM_model_eval.ipynb           # Support Vector Machine
│   ├── 5b-SVM_model_eval_OPTUNA.ipynb   # SVM com Optuna
│   ├── 6-XGBoost_model_eval.ipynb       # XGBoost
│   ├── 7-LightGBM_model_eval.ipynb      # LightGBM
│   ├── 8-MLP_model_eval.ipynb           # Multi-Layer Perceptron
│   ├── 9-Stacking_model_eval.ipynb      # Stacking Ensemble
│   ├── 10-Neural_Committee_model_eval.ipynb  # Comitê de Redes Neurais
│   ├── plot_results.ipynb               # Comparação de resultados
│   ├── plot_roc_curves.ipynb            # Geração de curvas ROC
│   ├── manual_implement_models.py       # Modelos customizados
│   ├── ml_utils.py                      # Utilidades de ML
│   ├── search_utils.py                  # RandomizedSearchCV utilities
│   ├── search_utils_optuna.py           # Optuna optimization utilities
│   ├── 📁 results/                      # Resultados em JSON
│   ├── 📁 searches/                     # Histórico de buscas
│   └── 📁 models/                       # Modelos treinados salvos
│
├── 📁 img/                               # Imagens e gráficos
│
├── 1-eda-sepsis.ipynb                   # Análise Exploratória de Dados
├── 2-data-prep-sepsis-v2.ipynb          # Pré-processamento (versão final)
├── 3-model_eval.ipynb                   # Avaliação inicial de modelos
│
├── dataset_sepsis.csv                   # Dataset original completo
├── dataset_sepsis_train.csv             # Dataset de treino (sem prep)
├── dataset_sepsis_test.csv              # Dataset de teste (sem prep)
├── dataset_sepsis_train_pid_prep_v2.csv # Treino pré-processado (v2)
├── dataset_sepsis_test_pid_prep_v2.csv  # Teste pré-processado (v2)
│
├── split_dataset.py                     # Script de divisão treino/teste
├── eda-sepsis.py                        # EDA em script Python
├── requirements.txt                     # Dependências do projeto
├── LICENSE                              # Licença MIT
└── README.md                            # Este arquivo
```

---

## 🔄 Metodologia CRISP-DM

Este projeto segue rigorosamente as 6 fases da metodologia **CRISP-DM** (Cross-Industry Standard Process for Data Mining):

### 1. 📌 Business Understanding
**Objetivo**: Desenvolver um sistema de predição precoce de sepse para auxiliar médicos na tomada de decisão clínica, reduzindo mortalidade e custos hospitalares.

- **Problema**: Sepse é uma das principais causas de morte em UTIs (~30% mortalidade)
- **Meta**: Construir modelo preditivo com alta sensibilidade (recall) para detecção precoce
- **Métrica primária**: F1-Score (balanceamento entre precisão e recall)
- **Restrições**: Dataset altamente desbalanceado, presença significativa de missing values

### 2. 📊 Data Understanding
**Notebooks**: `1-eda-sepsis.ipynb`

- Análise estatística descritiva completa
- Visualização de distribuições e correlações
- Análise de missing values (até 96% em algumas variáveis)
- Identificação de padrões temporais
- Análise de desbalanceamento de classes

### 3. 🔧 Data Preparation
**Notebooks**: `2-data-prep-sepsis-v2.ipynb`

#### Pipeline de Pré-processamento:

1. **Criação de Patient ID**: Identificação única baseada em Hour=0 + mudança de Age
2. **Imputação por Paciente**: Forward/Backward fill temporal preservando continuidade clínica
3. **Seleção de Variáveis**:
   - **Análise de Separabilidade**: Separabilidade = |mediana_sepsis - mediana_não_sepsis| / std_pooled
   - **Teste Mann-Whitney**: Significância estatística (p < 0.05)
   - **Missing Threshold**: Descarte de variáveis com >60% missing + baixa separabilidade
   - **Redundância**: Remoção de SBP/DBP (mantido MAP)
   
4. **Transformações de Normalidade**:
   - **Platelets**: Box-Cox (λ ≈ 0.3)
   - **WBC**: Yeo-Johnson (λ ≈ 0.8)
   - **BUN, MAP, Creatinine, Glucose**: Logaritmo natural
   
5. **Balanceamento**: Undersampling da classe majoritária (5% dos Não-Sepsis)
6. **Normalização**: StandardScaler (Z-score) em todas as features numéricas

#### Variáveis Finais Selecionadas:
- Sinais Vitais: HR, O2Sat, Temp, Resp, MAP
- Exames: BUN, Creatinine, Glucose, Hct, Hgb, WBC, Platelets
- Temporal: Hour, ICULOS, HospAdmTime
- Demográfica: Gender

### 4. 🤖 Modeling
**Notebooks**: `modeling/1-*.ipynb` até `modeling/10-*.ipynb`

#### Modelos Implementados:

| Categoria | Modelos |
|-----------|---------|
| **Baseados em Instância** | KNN, LVQ (Learning Vector Quantization) |
| **Baseados em Árvore** | Decision Tree, Random Forest, XGBoost, LightGBM |
| **Kernel Methods** | SVM (Linear, RBF, Polynomial) |
| **Redes Neurais** | MLP (Multi-Layer Perceptron) |
| **Ensemble Avançado** | Stacking Heterogêneo, Comitê de Redes Neurais |

#### Estratégia de Otimização:

- **RandomizedSearchCV**: 20 buscas × 80 iterações cada (base)
- **Optuna**: Busca Bayesiana para SVM (experimento)
- **Cross-Validation**: 5-fold Stratified CV
- **Métrica de Otimização**: F1-Score macro
- **Amostragem**: 5% do dataset de treino para acelerar buscas

### 5. 📈 Evaluation
**Notebooks**: `modeling/plot_results.ipynb`, `modeling/plot_roc_curves.ipynb`

#### Métricas Avaliadas:

- **F1-Score**: Métrica primária (balanceamento precisão/recall)
- **Precision & Recall**: Análise de trade-offs
- **G-Mean**: √(Sensitivity × Specificity) para dados desbalanceados
- **AUC-ROC**: Capacidade discriminativa geral
- **Confusion Matrix**: Análise de erros tipo I e II
- **Youden's Index**: Melhor threshold da curva ROC

#### Visualizações:

- Curvas ROC com probabilidades preditas reais
- Comparação de métricas entre modelos
- Análise de overfitting (treino vs validação)
- Distribuição de probabilidades por classe
- Matrizes de confusão normalizadas

### 6. 🚀 Deployment
**Status**: Projeto acadêmico finalizado

- Todos os modelos exportados em formato `.pkl` (joblib)
- Resultados salvos em JSON para reprodutibilidade
- Pipeline de pré-processamento documentado
- Código modular e reutilizável

---

## 🤖 Modelos Implementados

### 1. K-Nearest Neighbors (KNN)
- **Hiperparâmetros**: n_neighbors, metric (euclidean/manhattan), weights (uniform/distance)
- **Resultado**: F1-Score ~0.59, alto overfitting (treino ≈1.0, validação ~0.35-0.60)
- **Observação**: Extrema sensibilidade a k, memorização do dataset

### 2. Learning Vector Quantization (LVQ)
- **Implementação**: Customizada (classe LVQClassifier)
- **Hiperparâmetros**: prototypes_per_class, n_epochs, learning_rate
- **Resultado**: Desempenho moderado, boa interpretabilidade

### 3. Decision Tree
- **Hiperparâmetros**: max_depth, min_samples_split, min_samples_leaf, criterion
- **Resultado**: F1-Score ~0.53, quedas abruptas em validação
- **Observação**: Configurações específicas causam árvores excessivamente complexas/simples

### 4. Random Forest ⭐
- **Hiperparâmetros**: n_estimators, max_depth, min_samples_split, max_features
- **Resultado**: **Melhor desempenho geral** - F1-Score ~0.60 (±0.012)
- **Observação**: Validação estável apesar de oscilações no treino (robustez do ensemble)

### 5. Support Vector Machine (SVM)
- **Kernels testados**: Linear, RBF, Polynomial
- **Hiperparâmetros**: C, gamma, degree
- **Resultado**: F1-Score ~0.52, extrema sensibilidade aos hiperparâmetros
- **Observação**: Janela estreita de configurações ótimas

### 6. XGBoost
- **Hiperparâmetros**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- **Resultado**: F1-Score ~0.55, overfitting visível (treino ~0.9, validação ~0.5)
- **Observação**: Exact greedy algorithm se ajusta fortemente aos dados de treino

### 7. LightGBM
- **Hiperparâmetros**: Similar ao XGBoost + num_leaves, min_child_samples
- **Resultado**: F1-Score ~0.55, oscilações pronunciadas
- **Observação**: Leaf-wise growth gera picos mais acentuados que XGBoost

### 8. Multi-Layer Perceptron (MLP)
- **Arquiteturas**: 1-3 camadas ocultas, 50-200 neurônios
- **Hiperparâmetros**: hidden_layer_sizes, alpha, learning_rate_init, activation
- **Resultado**: F1-Score ~0.54, requer ajuste fino de regularização

### 9. Stacking Ensemble
- **Implementação**: Classe HeterogeneousStackingCommittee
- **Base Learners**: Decision Tree (shallow) + MLP (weak) + XGBoost (conservador)
- **Meta-Learner**: Logistic Regression (C=1-50)
- **Resultado**: F1-Score ~0.56, combina forças de paradigmas diferentes

### 10. Neural Committee
- **Implementação**: Classe NeuralNetworkCommittee (VotingClassifier de 3 MLPs)
- **Arquiteturas**: MLP1 (relu), MLP2 (tanh), MLP3 (logistic) com configs individuais
- **Votação**: Soft voting (predict_proba)
- **Resultado**: F1-Score ~0.55, ensemble homogêneo de redes neurais

---

### Observações Principais

1. **Random Forest** dominou pela estabilidade (menor desvio padrão) e robustez do ensemble
2. **KNN** surpreendeu com alto F1-Score mas com **overfitting severo** (treino ≈1.0)
3. **Gradient Boosting** (XGBoost, LightGBM) teve desempenho similar mas com maior overfitting
4. **Ensembles avançados** (Stacking, Neural Committee) não superaram Random Forest neste dataset
5. **Todos os modelos** lutaram com a natureza desbalanceada e missing values do dataset

---

## 💻 Instalação e Uso

### Requisitos

- Python 3.8+
- 8GB RAM mínimo (16GB recomendado para modelos ensemble)

### Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/data-mining-proj-crisp-dm.git
cd data-mining-proj-crisp-dm

# 2. Criar ambiente virtual
python -m venv .venv

# 3. Ativar ambiente virtual
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 4. Instalar dependências
pip install -r requirements.txt
```

### Executar Pipeline Completo

```bash
# 1. Dividir dataset (se necessário)
python split_dataset.py

# 2. Executar EDA (opcional)
jupyter notebook 1-eda-sepsis.ipynb

# 3. Pré-processar dados
jupyter notebook 2-data-prep-sepsis-v2.ipynb

# 4. Treinar modelos individuais
jupyter notebook modeling/4-RandomForest_model_eval.ipynb

# 5. Gerar curvas ROC
jupyter notebook modeling/plot_roc_curves.ipynb

# 6. Comparar resultados
jupyter notebook modeling/plot_results.ipynb
```

### Uso Rápido - Carregar Modelo Treinado

```python
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carregar modelo
model = joblib.load('modeling/models/random_forest_best_model.joblib')

# Carregar dados de teste pré-processados
X_test = pd.read_csv('dataset_sepsis_test_pid_prep_v2.csv')
y_test = X_test['SepsisLabel']
X_test = X_test.drop('SepsisLabel', axis=1)

# Predição
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Avaliação
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

## 📚 Dicionário de Variáveis

### 🕒 Identificadores e Tempo
- **Hour**: Hora desde a admissão na UTI (0-335+ horas)
- **ICULOS**: Duração da estadia na UTI em horas
- **HospAdmTime**: Tempo entre admissão hospitalar e UTI (horas, negativos = admissão direta)
- **PATIENT_ID** (criado): Identificador único de paciente

### ❤️ Sinais Vitais (Vital Signs)
- **HR**: Frequência cardíaca (batimentos/min)
- **O2Sat**: Saturação de oxigênio periférico (%)
- **Temp**: Temperatura corporal (°C)
- **SBP**: Pressão arterial sistólica (mmHg) [removida - redundante com MAP]
- **MAP**: Pressão arterial média (mmHg) ⭐ **Mantida**
- **DBP**: Pressão arterial diastólica (mmHg) [removida - redundante com MAP]
- **Resp**: Taxa respiratória (respirações/min)

### 🫁 Gases Sanguíneos (Blood Gases) [Maioria removida por >60% missing]
- ~~EtCO2~~: CO2 expirado
- ~~BaseExcess~~: Excesso de base
- ~~HCO3~~: Bicarbonato
- ~~FiO2~~: Fração inspirada de oxigênio
- ~~pH~~: pH arterial
- ~~PaCO2~~: Pressão parcial de CO2
- ~~SaO2~~: Saturação de oxigênio arterial

### 🧪 Exames Laboratoriais (Laboratory Tests)
**Mantidos (boa separabilidade)**:
- **BUN**: Ureia (mg/dL) - Função renal
- **Creatinine**: Creatinina (mg/dL) - Função renal ⭐
- **Glucose**: Glicose (mg/dL) - Metabolismo

**Removidos (alta missing + baixa separabilidade)**:
- ~~AST~~: Aspartato aminotransferase
- ~~Alkalinephos~~: Fosfatase alcalina
- ~~Calcium~~: Cálcio
- ~~Chloride~~: Cloreto
- ~~Bilirubin_direct~~: Bilirrubina direta
- ~~Lactate~~: Lactato [surpreendentemente baixa separabilidade]
- ~~Magnesium~~: Magnésio
- ~~Phosphate~~: Fosfato
- ~~Potassium~~: Potássio
- ~~Bilirubin_total~~: Bilirrubina total
- ~~TroponinI~~: Troponina I

### 🩸 Hematologia (Hematology)
**Mantidos**:
- **Hct**: Hematócrito (%) - Volume de hemácias
- **Hgb**: Hemoglobina (g/dL) - Capacidade de oxigenação
- **WBC**: Contagem de leucócitos (1000/uL) - Infecção/inflamação ⭐
- **Platelets**: Contagem de plaquetas (1000/uL) - Coagulação ⭐

**Removidos**:
- ~~PTT~~: Tempo de tromboplastina parcial
- ~~Fibrinogen~~: Fibrinogênio

### 👤 Informações Demográficas
- **Age**: Idade (anos) [removida - baixa separabilidade]
- **Gender**: Gênero (0=Feminino, 1=Masculino) ⭐ **Mantida**
- ~~Unit1~~: UTI Médica [removida - >90% missing]
- ~~Unit2~~: UTI Cirúrgica/Cardiológica [removida - >90% missing]

### 🎯 Variável Alvo
- **SepsisLabel**: Rótulo de sepse (0=Não-Sepsis, 1=Sepsis)


## 📖 Referências

1. **PhysioNet 2019 Challenge**: [https://physionet.org/content/challenge-2019/](https://physionet.org/content/challenge-2019/)
2. **Sepsis-3 Definition**: Singer M, et al. JAMA. 2016;315(8):801-810
3. **CRISP-DM Methodology**: [https://www.datascience-pm.com/crisp-dm-2/](https://www.datascience-pm.com/crisp-dm-2/)
4. **Scikit-learn Documentation**: [https://scikit-learn.org/](https://scikit-learn.org/)
5. **Imbalanced-learn**: [https://imbalanced-learn.org/](https://imbalanced-learn.org/)

---

## 📝 Observações Finais

Este projeto está **oficialmente finalizado** (Dezembro 2025). Atualizações esporádicas podem ocorrer para:
- Experimentação de novos modelos
- Otimizações de hiperparâmetros
- Melhorias na documentação
- Correções de bugs

Para questões ou sugestões, abra uma **issue** no GitHub.
