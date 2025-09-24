"""
Script para dividir o dataset de sepsis entre dados de treino e teste.
Possui uma etapa que corrige o problema de estratificação removendo valores 
NaN da target antes da divisão.
PS: Apenas 34 linhas removidas de mais de 1.5M
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_sepsis_dataset(file_path):
    """
    Carrega o dataset de sepsis a partir do arquivo.
    
    Args:
        file_path (str): Caminho para o arquivo do dataset
        
    Returns:
        pandas.DataFrame: Dataset carregado
    """
    print("Carregando dataset...")
    
    # Primeiro, conta quantas linhas são comentários no início
    comment_lines = 0
    data_lines_sample = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.startswith('%') or line.startswith('@'):
                comment_lines += 1
            elif line and ',' in line:  # Linha com dados (contém vírgulas)
                data_lines_sample.append(line)
                if len(data_lines_sample) >= 10:  # Amostra de 10 linhas para análise
                    break
    
    print(f"Linhas de comentário/header: {comment_lines}")
    
    # Analisa a estrutura das linhas de dados
    if not data_lines_sample:
        raise ValueError("Não foi possível encontrar linhas de dados no arquivo")
    
    # Determina o número de colunas baseado na linha mais comum
    col_counts = {}
    for line in data_lines_sample:
        num_cols = len(line.split(','))
        col_counts[num_cols] = col_counts.get(num_cols, 0) + 1
    
    # Usa o número de colunas mais comum
    num_cols = max(col_counts, key=col_counts.get)
    print(f"Número de colunas detectadas: {num_cols}")
    
    # Nomes das colunas baseados na documentação do PhysioNet 2019 Challenge
    feature_names = [
        'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
        'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
        'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
        'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
        'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
        'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
    ]
    
    # Se temos mais colunas que nomes, adiciona nomes genéricos
    if num_cols > len(feature_names) + 1:
        for i in range(len(feature_names), num_cols - 1):
            feature_names.append(f'Feature_{i}')
    
    # Adiciona a coluna target (SepsisLabel)
    column_names = feature_names[:num_cols-1] + ['SepsisLabel']
    
    print(f"Colunas: {column_names[:5]}...{column_names[-3:]}")
    
    # Carrega o dataset usando pandas diretamente
    try:
        df = pd.read_csv(file_path, 
                        skiprows=comment_lines,
                        names=column_names,
                        na_values=['?', '', ' '],
                        low_memory=False)
        
        print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
    except Exception as e:
        print(f"Erro ao carregar: {e}")
        raise
    
    return df

def clean_and_split_dataset(df, test_size=0.2, random_state=42, output_dir='.'):
    """
    Limpa o dataset e divide em treino e teste com estratificação adequada.
    
    Args:
        df (pandas.DataFrame): Dataset completo
        test_size (float): Proporção dos dados para teste (padrão: 20%)
        random_state (int): Seed para reprodutibilidade
        output_dir (str): Diretório onde salvar os arquivos
    """
    print(f"\n=== LIMPEZA E DIVISÃO DO DATASET ===")
    print(f"Dataset original: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Separa features e target
    X = df.drop('SepsisLabel', axis=1)
    y = df['SepsisLabel']
    
    # Verifica valores NaN na target
    target_na_count = y.isna().sum()
    target_na_pct = (target_na_count / len(y)) * 100
    print(f"Valores NaN na target: {target_na_count} ({target_na_pct:.4f}%)")
    
    # Remove linhas com target NaN para permitir estratificação
    if target_na_count > 0:
        print(f"Removendo {target_na_count} linhas com target NaN...")
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        print(f"Dataset após limpeza: {len(y)} linhas")
    
    # Verifica distribuição da target
    print(f"\nDistribuição da target (após limpeza):")
    target_dist = y.value_counts().sort_index()
    target_pct = y.value_counts(normalize=True).sort_index()
    
    for label in sorted(y.unique()):
        count = target_dist[label]
        pct = target_pct[label]
        label_name = "Não-Sepsis" if label == 0 else "Sepsis"
        print(f"  {label_name} ({label}): {count} ({pct:.1%})")
    
    # Verifica se é possível fazer estratificação
    min_class_count = target_dist.min()
    min_test_samples = int(min_class_count * test_size)
    
    print(f"\nVerificação para estratificação:")
    print(f"  Menor classe: {min_class_count} amostras")
    print(f"  Amostras mínimas para teste: {min_test_samples}")
    
    # Divisão estratificada
    try:
        if min_test_samples >= 1:  # Precisa de pelo menos 1 amostra de cada classe no teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y
            )
            print("Divisão estratificada realizada")
            stratified = True
        else:
            raise ValueError("Classe minoritária muito pequena para estratificação")
            
    except Exception as e:
        print(f"❌ Estratificação falhou: {e}")
        print("Realizando divisão aleatória...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        stratified = False
    
    # Reconstrói os DataFrames completos
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Salva os datasets
    train_path = os.path.join(output_dir, 'dataset_sepsis_train.csv')
    test_path = os.path.join(output_dir, 'dataset_sepsis_test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n=== RESULTADOS ===")
    print(f"Arquivos salvos:")
    print(f"  Treino: {train_path} ({train_df.shape[0]} amostras)")
    print(f"  Teste: {test_path} ({test_df.shape[0]} amostras)")
    
    # Estatísticas da divisão
    print(f"\nEstatísticas da divisão:")
    print(f"  Total: {len(y)} amostras")
    print(f"  Treino: {len(y_train)} ({len(y_train)/len(y)*100:.1f}%)")
    print(f"  Teste: {len(y_test)} ({len(y_test)/len(y)*100:.1f}%)")
    print(f"  Estratificada: {'Sim' if stratified else 'Não'}")
    
    # Verifica se a distribuição foi mantida
    if stratified:
        print(f"\nVerificação da estratificação:")
        
        print("  Dataset completo:")
        for label in sorted(y.unique()):
            pct = (y == label).mean()
            print(f"    Classe {label}: {pct:.1%}")
            
        print("  Treino:")
        for label in sorted(y_train.unique()):
            pct = (y_train == label).mean()
            print(f"    Classe {label}: {pct:.1%}")
            
        print("  Teste:")
        for label in sorted(y_test.unique()):
            pct = (y_test == label).mean()
            print(f"    Classe {label}: {pct:.1%}")
    
    return train_df, test_df

def main():
    # Configurações
    dataset_path = 'dataset_sepsis'
    output_directory = '.'
    test_proportion = 0.2  # 20% para teste, 80% para treino
    random_seed = 42  # Para reprodutibilidade
    
    print(f"Arquivo de entrada: {dataset_path}")
    print(f"Proporção de teste: {test_proportion*100}%")
    print(f"Seed aleatória: {random_seed}")
    
    try:
        # Carrega o dataset
        df = load_sepsis_dataset(dataset_path)
        
        # Limpa e divide
        train_df, test_df = clean_and_split_dataset(
            df, 
            test_size=test_proportion,
            random_state=random_seed,
            output_dir=output_directory
        )
        
        print("\n dividido com sucesso")
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()