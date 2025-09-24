# Data Mining Project - CRISP-DM Methodology
## Sepsis Prediction using PhysioNet 2019 Challenge Dataset

Este projeto aplica a metodologia CRISP-DM para desenvolvimento de um modelo de predição de sepse utilizando dados clínicos de UTI.

## Dataset

O dataset utilizado é do PhysioNet 2019 Challenge, contendo dados de pacientes internados em UTI com múltiplas variáveis clínicas coletadas ao longo do tempo.

## Dicionário de Variáveis

### Identificadores e Tempo
- **Hour**: Hora desde a admissão na UTI (0-335+ horas)
- **ICULOS**: Duração da estadia na UTI em horas

### Sinais Vitais
- **HR**: Frequência cardíaca (batimentos por minuto)
- **O2Sat**: Saturação de oxigênio periférico (%)
- **Temp**: Temperatura corporal (°C)
- **SBP**: Pressão arterial sistólica (mmHg)
- **MAP**: Pressão arterial média (mmHg)
- **DBP**: Pressão arterial diastólica (mmHg)
- **Resp**: Taxa respiratória (respirações por minuto)

### Gases Sanguíneos
- **EtCO2**: CO2 expirado (mmHg)
- **BaseExcess**: Excesso de base (mmol/L)
- **HCO3**: Bicarbonato (mmol/L)
- **FiO2**: Fração inspirada de oxigênio (0.0-1.0)
- **pH**: pH arterial
- **PaCO2**: Pressão parcial de CO2 arterial (mmHg)
- **SaO2**: Saturação de oxigênio arterial (%)

### Exames Laboratoriais
- **AST**: Aspartato aminotransferase (IU/L)
- **BUN**: Ureia (mg/dL)
- **Alkalinephos**: Fosfatase alcalina (IU/L)
- **Calcium**: Cálcio (mg/dL)
- **Chloride**: Cloreto (mmol/L)
- **Creatinine**: Creatinina (mg/dL)
- **Bilirubin_direct**: Bilirrubina direta (mg/dL)
- **Glucose**: Glicose (mg/dL)
- **Lactate**: Lactato (mmol/L)
- **Magnesium**: Magnésio (mmol/L)
- **Phosphate**: Fosfato (mg/dL)
- **Potassium**: Potássio (mmol/L)
- **Bilirubin_total**: Bilirrubina total (mg/dL)
- **TroponinI**: Troponina I (ng/mL)

### Hematologia
- **Hct**: Hematócrito (%)
- **Hgb**: Hemoglobina (g/dL)
- **PTT**: Tempo de tromboplastina parcial (segundos)
- **WBC**: Contagem de glóbulos brancos (1000/uL)
- **Fibrinogen**: Fibrinogênio (mg/dL)
- **Platelets**: Contagem de plaquetas (1000/uL)

### Informações Demográficas e Administrativas
- **Age**: Idade do paciente (anos)
- **Gender**: Gênero do paciente (0=Feminino, 1=Masculino)
- **Unit1**: Tipo de UTI - Unidade Médica (0=Não, 1=Sim)
- **Unit2**: Tipo de UTI - Unidade Cirúrgica/Cardiológica (0=Não, 1=Sim)
- **HospAdmTime**: Tempo entre admissão hospitalar e UTI (horas, valores negativos = admissão direta)

### Variável Alvo
- **SepsisLabel**: Rótulo de sepsis (0=Não-Sepsis, 1=Sepsis)

## Informações Importantes sobre Unit1 e Unit2

As variáveis Unit1 e Unit2 são binárias mutuamente exclusivas que representam o tipo de UTI:

- **Unit1 (UTI Médica)**: Pacientes com condições clínicas agudas
  - Condições típicas: pneumonia grave, sepse, insuficiência respiratória
  - Perfil: pacientes geralmente mais instáveis, maior mortalidade
  - Foco: tratamento de doenças médicas agudas

- **Unit2 (UTI Cirúrgica/Cardiológica)**: Pacientes pós-operatórios ou com problemas cardíacos
  - Condições típicas: pós-cirurgia cardíaca, grandes cirurgias abdominais
  - Perfil: monitorização intensiva durante recuperação
  - Foco: estabilização pós-procedimento

**Características importantes:**
- Unit1 + Unit2 = 1 para todos os pacientes
- Representam diferentes ambientes clínicos com protocolos distintos
- Importantes para estratificação de risco e feature engineering

## Características do Dataset

- **Distribuição de classes**: Altamente desbalanceado (~98.2% Não-Sepsis, 1.8% Sepsis)
- **Dados faltantes**: Presença significativa de valores ausentes (NaN) em várias variáveis clínicas
- **Dados temporais**: Múltiplas medições por paciente ao longo do tempo
- **Critério de sepse**: Baseado na definição Sepsis-3

## Estrutura do Projeto

```
data-minning-proj-crisp-dm/
├── dataset_sepsis/                 # Dataset original
├── dataset_sepsis_train.csv        # Dados de treino (80%)
├── dataset_sepsis_test.csv         # Dados de teste (20%)
├── eda-sepsis.py                   # Análise exploratória estruturada
├── split_dataset.py                # Script de divisão do dataset
├── split_dataset_fixed.py          # Versão corrigida com tratamento de NaN
├── split_for_repo.py              # Divisão para compatibilidade GitHub
├── load_train_dataset.py          # Utilitário para carregar dados divididos
└── README.md                      # Documentação do projeto
```

## Scripts Principais

### eda-sepsis.py
Script principal de análise exploratória seguindo estrutura organizada:
- **Tarefa 2**: O Detetive de NaNs - Análise quantitativa e visual dos dados faltantes
- **Tarefa 3**: Desvendando as Categorias - Análise de variáveis categóricas
- **Tarefa 4**: Conectando Categorias e Números - Análise mista categórica vs numérica

### split_dataset_fixed.py
Script de divisão estratificada do dataset com tratamento adequado de valores ausentes na variável alvo.

## Metodologia CRISP-DM

Este projeto segue as fases da metodologia CRISP-DM:

1. **Business Understanding**: Predição precoce de sepse para melhoria de desfechos clínicos
2. **Data Understanding**: Análise exploratória completa do dataset PhysioNet 2019
3. **Data Preparation**: Limpeza, tratamento de NaNs e feature engineering
4. **Modeling**: Desenvolvimento de modelos de machine learning
5. **Evaluation**: Avaliação com métricas apropriadas para dados desbalanceados
6. **Deployment**: Preparação para implementação clínica

## Dependências

```python
pandas
numpy
scikit-learn
matplotlib
seaborn
warnings
```

## Como Executar

1. **Carregar e dividir o dataset:**
```bash
python split_dataset.py
```

2. **Executar análise exploratória:**
```bash
python eda-sepsis.py
```

## Observações Técnicas

- **Estratificação**: Divisão treino/teste mantém proporção das classes
- **Tratamento de NaN**: Remoção de registros com target ausente antes da divisão
- **Compatibilidade GitHub**: Arquivos grandes divididos em partes menores
- **Reprodutibilidade**: Random state fixo para resultados consistentes

## Próximos Passos

1. Pré-processamento avançado dos dados
2. Feature engineering baseada em conhecimento clínico
3. Desenvolvimento de modelos de machine learning
4. Otimização de hiperparâmetros
5. Validação com métricas clínicas relevantes