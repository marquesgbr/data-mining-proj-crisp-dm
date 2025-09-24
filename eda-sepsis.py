"""
Análise Exploratória de Dados (EDA) - Dataset de Sepsis
Assume que os dados de treino e teste já foram divididos e estão disponíveis como:
'dataset_sepsis_train.csv' e 'dataset_sepsis_test.csv'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuração de plotagem
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def load_and_prepare_data():
    """Carrega e prepara os dados para análise (assumindo divisão já feita)."""
    print("Carregando dataset de treino...")
    train_df = pd.read_csv('dataset_sepsis_train.csv')
    
    # Separar X e y para análise
    X_train = train_df.drop('SepsisLabel', axis=1)
    y_train = train_df['SepsisLabel']
    
    print(f"Dados carregados - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Distribuição target: {y_train.value_counts().to_dict()}")
    
    return X_train, y_train

def detective_nans(X_train):
    """
    TAREFA 2: O Detetive de NaNs
    Analisar e quantificar os NaNs, visualizar padrões e anotar observações.
    """
    print("\n=== O DETETIVE DE NANS ===")
    
    # Analisar e quantificar os NaNs
    print("Análise quantitativa dos NaNs:")
    missing_count = X_train.isnull().sum()
    missing_pct = (missing_count / len(X_train)) * 100
    
    # Cria DataFrame para análise
    missing_df = pd.DataFrame({
        'Coluna': missing_count.index,
        'Valores_Faltantes': missing_count.values,
        'Porcentagem': missing_pct.values
    }).sort_values('Porcentagem', ascending=False)
    
    # Remove colunas sem NaNs para o relatório
    missing_with_nans = missing_df[missing_df['Valores_Faltantes'] > 0]
    
    print(f"\nResumo dos NaNs:")
    print(f"   Total de colunas: {len(X_train.columns)}")
    print(f"   Colunas com NaNs: {len(missing_with_nans)}")
    print(f"   Colunas sem NaNs: {len(X_train.columns) - len(missing_with_nans)}")
    
    print(f"\nTop 10 colunas com mais NaNs:")
    print(missing_with_nans.head(10).to_string(index=False))
    
    # Visualizar os NaNs com heatmap
    print(f"\nGerando heatmap dos NaNs...")
    plt.figure(figsize=(15, 10))
    
    # Seleciona subset de colunas para visualização
    cols_with_nans = missing_with_nans.head(20)['Coluna'].tolist()
    
    if len(cols_with_nans) > 0:
        plt.subplot(2, 1, 1)
        sns.heatmap(X_train[cols_with_nans].isnull(), 
                   cbar=True, 
                   yticklabels=False,
                   cmap='viridis')
        plt.title('Padrão de Valores Faltantes (Top 20 colunas com NaNs)')
        plt.xticks(rotation=45)
        
        # Gráfico de barras com percentuais
        plt.subplot(2, 1, 2)
        top_missing = missing_with_nans.head(15)
        plt.barh(range(len(top_missing)), top_missing['Porcentagem'])
        plt.yticks(range(len(top_missing)), top_missing['Coluna'])
        plt.xlabel('Porcentagem de NaNs (%)')
        plt.title('Top 15 Colunas com Maior % de NaNs')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('nans_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("   Nenhum NaN encontrado no dataset!")
    
    # Anotações
    print(f"\n=== ANOTAÇÕES DOS NANs ===")
    print(f"Quais colunas têm NaNs?")
    if len(missing_with_nans) > 0:
        print(f"   As {len(missing_with_nans)} colunas com NaNs são:")
        for _, row in missing_with_nans.head(10).iterrows():
            print(f"   • {row['Coluna']}: {row['Porcentagem']:.1f}%")
    
    print(f"\nQual a quantidade?")
    print(f"   • Total de valores faltantes: {missing_count.sum():,}")
    print(f"   • % do dataset: {(missing_count.sum() / (len(X_train) * len(X_train.columns)))*100:.2f}%")
    
    print(f"\nHá algum padrão?")
    if len(missing_with_nans) > 0:
        high_missing = missing_with_nans[missing_with_nans['Porcentagem'] > 80]
        medium_missing = missing_with_nans[(missing_with_nans['Porcentagem'] > 20) & (missing_with_nans['Porcentagem'] <= 80)]
        low_missing = missing_with_nans[missing_with_nans['Porcentagem'] <= 20]
        
        print(f"   • Muito missing (>80%): {len(high_missing)} colunas")
        print(f"   • Médio missing (20-80%): {len(medium_missing)} colunas")  
        print(f"   • Baixo missing (<20%): {len(low_missing)} colunas")
    
    print(f"\nPrimeira ideia de tratamento:")
    if len(missing_with_nans) > 0:
        print(f"   • Colunas >80% NaN: Considerar remoção")
        print(f"   • Colunas 20-80% NaN: Imputação específica por tipo")
        print(f"   • Colunas <20% NaN: Imputação simples (mediana/moda)")
    else:
        print(f"   • Dataset já está limpo, sem necessidade de tratamento!")
    
    return missing_df

def desvendando_categorias(X_train, y_train):
    """
    TAREFA 3: Desvendando as Categorias
    Escolher 2-3 variáveis categóricas e criar gráficos simples e vs target.
    """
    print("\n=== DESVENDANDO AS CATEGORIAS ===")
    
    # Identificar variáveis categóricas
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Para dataset de sepsis, algumas variáveis numéricas são categóricas
    potential_categorical = []
    for col in X_train.columns:
        if X_train[col].nunique() <= 10 and X_train[col].dtype in ['int64', 'float64']:
            potential_categorical.append(col)
    
    print(f"Variáveis categóricas identificadas:")
    print(f"   Categóricas explícitas: {len(categorical_cols)} -> {categorical_cols}")
    print(f"   Potencialmente categóricas (≤10 valores únicos): {len(potential_categorical)}")
    print(f"   {potential_categorical[:10]}{'...' if len(potential_categorical) > 10 else ''}")
    
    # Escolher 2-3 variáveis categóricas relevantes
    selected_categorical = []
    
    # Priorizar algumas variáveis conhecidas do dataset de sepsis
    important_categorical = ['Gender', 'Unit1', 'Unit2', 'Hour']
    
    for col in important_categorical:
        if col in X_train.columns and X_train[col].nunique() <= 20:
            selected_categorical.append(col)
        if len(selected_categorical) >= 3:
            break
    
    # Se não encontrou suficientes, pega das potencialmente categóricas
    if len(selected_categorical) < 3:
        for col in potential_categorical:
            if col not in selected_categorical:
                selected_categorical.append(col)
            if len(selected_categorical) >= 3:
                break
    
    print(f"\nVariáveis selecionadas para análise: {selected_categorical}")
    
    # Criar gráficos categóricos
    fig, axes = plt.subplots(len(selected_categorical), 2, figsize=(15, 5*len(selected_categorical)))
    if len(selected_categorical) == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(selected_categorical):
        # Gráfico simples de contagem
        plt.subplot(len(selected_categorical), 2, 2*i + 1)
        counts = X_train[col].value_counts().head(10)
        sns.countplot(data=X_train, x=col, order=counts.index)
        plt.title(f'Distribuição de {col}')
        plt.xticks(rotation=45)
        
        # Gráfico Categoria vs. Variável Alvo
        plt.subplot(len(selected_categorical), 2, 2*i + 2)
        temp_df = pd.DataFrame({'cat': X_train[col], 'target': y_train})
        top_categories = X_train[col].value_counts().head(10).index
        temp_df_filtered = temp_df[temp_df['cat'].isin(top_categories)]
        
        sns.countplot(data=temp_df_filtered, x='cat', hue='target')
        plt.title(f'{col} vs SepsisLabel')
        plt.xticks(rotation=45)
        plt.legend(title='SepsisLabel', labels=['Não-Sepsis', 'Sepsis'])
    
    plt.tight_layout()
    plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Anotações
    print(f"\n=== ANOTAÇÕES DAS CATEGORIAS ===")
    for col in selected_categorical:
        print(f"\n{col}:")
        value_counts = X_train[col].value_counts()
        print(f"   • Valores únicos: {X_train[col].nunique()}")
        print(f"   • Categoria mais frequente: {value_counts.index[0]} ({value_counts.iloc[0]:,} ocorrências)")
        
        # Relação com target
        crosstab = pd.crosstab(X_train[col], y_train, normalize='index')
        if len(crosstab) > 0:
            # Encontra categoria com maior proporção de sepsis
            max_sepsis_cat = crosstab[1.0].idxmax() if 1.0 in crosstab.columns else "N/A"
            max_sepsis_pct = crosstab[1.0].max() if 1.0 in crosstab.columns else 0
            print(f"   • Categoria com maior risco de sepsis: {max_sepsis_cat} ({max_sepsis_pct:.1%})")
    
    return selected_categorical

def conectando_categorias_numeros(X_train, y_train, categorical_cols):
    """
    TAREFA 4: Conectando Categorias e Números
    Escolher uma variável categórica e uma numérica para gráficos mistos.
    """
    print("\n=== CONECTANDO CATEGORIAS E NÚMEROS ===")
    
    # Identificar variáveis numéricas relevantes
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remover colunas com muitos NaNs para análise mais limpa
    numeric_cols_clean = []
    for col in numeric_cols:
        missing_pct = X_train[col].isnull().sum() / len(X_train) * 100
        if missing_pct < 50:  # Menos de 50% de NaNs
            numeric_cols_clean.append(col)
    
    print(f"Variáveis numéricas disponíveis: {len(numeric_cols_clean)}")
    
    # Selecionar variáveis numéricas importantes (sinais vitais)
    important_numeric = ['HR', 'SBP', 'DBP', 'Temp', 'Resp', 'O2Sat', 'Age']
    selected_numeric = []
    
    for col in important_numeric:
        if col in numeric_cols_clean:
            selected_numeric.append(col)
        if len(selected_numeric) >= 3:
            break
    
    # Se não encontrou suficientes, pega outras
    if len(selected_numeric) < 3:
        for col in numeric_cols_clean:
            if col not in selected_numeric:
                selected_numeric.append(col)
            if len(selected_numeric) >= 3:
                break
    
    print(f"Variáveis numéricas selecionadas: {selected_numeric}")
    print(f"Variáveis categóricas disponíveis: {categorical_cols}")
    
    # Criar gráficos mistos para cada combinação
    n_combinations = min(3, len(categorical_cols) * len(selected_numeric))
    fig, axes = plt.subplots(n_combinations, 1, figsize=(12, 6*n_combinations))
    
    if n_combinations == 1:
        axes = [axes]
    
    combination_count = 0
    annotations = []
    
    for cat_col in categorical_cols[:3]:  # Máximo 3 categóricas
        for num_col in selected_numeric[:1]:  # 1 numérica por categórica
            if combination_count >= n_combinations:
                break
                
            plt.subplot(n_combinations, 1, combination_count + 1)
            
            # Preparar dados removendo NaNs
            temp_df = pd.DataFrame({
                'categorical': X_train[cat_col],
                'numerical': X_train[num_col],
                'target': y_train
            }).dropna()
            
            # Filtrar top categorias para visualização limpa
            top_categories = temp_df['categorical'].value_counts().head(8).index
            temp_df_filtered = temp_df[temp_df['categorical'].isin(top_categories)]
            
            if len(temp_df_filtered) > 0:
                sns.boxplot(data=temp_df_filtered, x='categorical', y='numerical', hue='target')
                plt.title(f'{num_col} por {cat_col} e SepsisLabel')
                plt.xticks(rotation=45)
                plt.legend(title='SepsisLabel', labels=['Não-Sepsis', 'Sepsis'])
                
                # Coleta dados para anotações
                stats_by_cat = temp_df_filtered.groupby('categorical')['numerical'].agg(['mean', 'std', 'min', 'max'])
                outliers_info = []
                
                # Detecta outliers usando IQR
                for cat in top_categories:
                    cat_data = temp_df_filtered[temp_df_filtered['categorical'] == cat]['numerical']
                    Q1 = cat_data.quantile(0.25)
                    Q3 = cat_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = cat_data[(cat_data < Q1 - 1.5*IQR) | (cat_data > Q3 + 1.5*IQR)]
                    if len(outliers) > 0:
                        outliers_info.append(f"{cat}: {len(outliers)} outliers")
                
                annotations.append({
                    'categorical': cat_col,
                    'numerical': num_col,
                    'stats': stats_by_cat,
                    'outliers': outliers_info
                })
            
            combination_count += 1
    
    plt.tight_layout()
    plt.savefig('categorical_numerical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Anotações
    print(f"\n=== ANOTAÇÕES CATEGORIAS vs NÚMEROS ===")
    for annotation in annotations:
        cat_col = annotation['categorical']
        num_col = annotation['numerical']
        stats = annotation['stats']
        outliers = annotation['outliers']
        
        print(f"\n{num_col} por {cat_col}:")
        print(f"   • Como a distribuição varia entre categorias:")
        
        if len(stats) > 0:
            # Categoria com maior média
            max_mean_cat = stats['mean'].idxmax()
            min_mean_cat = stats['mean'].idxmin()
            print(f"     - Maior média: {max_mean_cat} ({stats.loc[max_mean_cat, 'mean']:.2f})")
            print(f"     - Menor média: {min_mean_cat} ({stats.loc[min_mean_cat, 'mean']:.2f})")
            
            # Categoria com maior variabilidade
            max_std_cat = stats['std'].idxmax()
            print(f"     - Maior variabilidade: {max_std_cat} (std: {stats.loc[max_std_cat, 'std']:.2f})")
        
        print(f"   • Outliers significativos:")
        if outliers:
            for outlier_info in outliers:
                print(f"     - {outlier_info}")
        else:
            print(f"     - Nenhum outlier significativo detectado")
    
    return annotations

def analyze_features(df):
    """Análise básica das features numéricas."""
    print("\n=== ANÁLISE DAS FEATURES ===")
    
    # Features numéricas (excluindo SepsisLabel)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'SepsisLabel' in numeric_features:
        numeric_features.remove('SepsisLabel')
    
    print(f"Número de features numéricas: {len(numeric_features)}")
    
    # Informações sobre as features do dataset PhysioNet 2019
    print(f"\nTipos de features no dataset:")
    feature_groups = {
        'Sinais Vitais': ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp'],
        'Gases Sanguíneos': ['EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2'],
        'Exames Laboratoriais': ['AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 
                               'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 
                               'Potassium', 'Bilirubin_total', 'TroponinI'],
        'Hematologia': ['Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets'],
        'Informações Clínicas': ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'Hour']
    }
    
    for group_name, group_features in feature_groups.items():
        available_features = [f for f in group_features if f in df.columns]
        print(f"  {group_name}: {len(available_features)} features disponíveis")
        if available_features:
            print(f"    {available_features[:5]}{'...' if len(available_features) > 5 else ''}")
    
    # Estatísticas descritivas das features de sinais vitais
    vital_signs = [f for f in feature_groups['Sinais Vitais'] if f in df.columns]
    if vital_signs:
        print(f"\nEstatísticas dos Sinais Vitais:")
        print(df[vital_signs].describe().round(2))
    
    return numeric_features

def plot_target_distribution(df):
    """Visualiza a distribuição da variável target (SepsisLabel)."""
    plt.figure(figsize=(10, 6))
    
    # Gráfico de barras
    target_counts = df['SepsisLabel'].value_counts()
    plt.subplot(1, 2, 1)
    target_counts.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Distribuição da Variável SepsisLabel')
    plt.xlabel('Classes')
    plt.ylabel('Frequência')
    plt.xticks([0, 1], ['Não-Sepsis (0)', 'Sepsis (1)'], rotation=45)
    
    # Gráfico de pizza
    plt.subplot(1, 2, 2)
    target_pct = df['SepsisLabel'].value_counts(normalize=True)
    plt.pie(target_pct.values, labels=['Não-Sepsis', 'Sepsis'], 
            autopct='%1.1f%%', colors=['skyblue', 'salmon'])
    plt.title('Proporção das Classes')
    
    plt.tight_layout()
    plt.savefig('sepsis_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_missing_data(df):
    """Análise detalhada dos dados faltantes."""
    print("\n=== ANÁLISE DE DADOS FALTANTES ===")
    
    missing_summary = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing_summary / len(df)) * 100
    
    # Features com mais de 50% de dados faltantes
    high_missing = missing_pct[missing_pct > 50]
    print(f"\nFeatures com >50% de dados faltantes: {len(high_missing)}")
    if len(high_missing) > 0:
        print(high_missing.head())
    
    # Features com poucos dados faltantes (< 10%)
    low_missing = missing_pct[(missing_pct > 0) & (missing_pct < 10)]
    print(f"\nFeatures com <10% de dados faltantes: {len(low_missing)}")
    if len(low_missing) > 0:
        print(low_missing.head())

def sample_feature_analysis(df, numeric_features):
    """Análise de algumas features específicas."""
    print("\n=== ANÁLISE DE FEATURES ESPECÍFICAS ===")
    
    # Analisa as primeiras 5 features com menos dados faltantes
    features_to_analyze = []
    for feature in numeric_features:
        missing_pct = (df[feature].isnull().sum() / len(df)) * 100
        if missing_pct < 20:  # Menos de 20% de dados faltantes
            features_to_analyze.append(feature)
        if len(features_to_analyze) >= 5:
            break
    
    if features_to_analyze:
        print(f"Analisando features: {features_to_analyze}")
        
        # Estatísticas por classe
        for feature in features_to_analyze[:3]:  # Primeiras 3
            print(f"\n--- {feature} ---")
            stats_by_class = df.groupby('SepsisLabel')[feature].agg(['count', 'mean', 'std', 'min', 'max'])
            print(stats_by_class)

def main():
    """Executa as tarefas 2-4 de EDA estruturado."""
    print("ANÁLISE EXPLORATÓRIA ESTRUTURADA - DATASET DE SEPSIS")
    print("=" * 60)
    print("Executando tarefas 2-4 (assumindo que train_test_split já foi feito)")
    
    # Carrega e prepara os dados
    X_train, y_train = load_and_prepare_data()
    
    # TAREFA 2: O Detetive de NaNs
    missing_analysis = detective_nans(X_train)
    
    # TAREFA 3: Desvendando as Categorias
    categorical_vars = desvendando_categorias(X_train, y_train)
    
    # TAREFA 4: Conectando Categorias e Números
    mixed_analysis = conectando_categorias_numeros(X_train, y_train, categorical_vars)
    
    print("\n=== EDA ESTRUTURADO CONCLUÍDO ===")
    print("Gráficos salvos:")
    print("   • nans_analysis.png")
    print("   • categorical_analysis.png") 
    print("   • categorical_numerical_analysis.png")
    
    print("\nPróximos passos:")
    print("   1. Revisar as anotações geradas")
    print("   2. Decidir estratégias de pré-processamento")
    print("   3. Fazer feature engineering")
    print("   4. Preparar para modelagem")
    
    return {
        'missing_analysis': missing_analysis,
        'categorical_vars': categorical_vars,
        'mixed_analysis': mixed_analysis
    }

if __name__ == "__main__":
    main()