# ==============================================================================
# CHECKPOINT 6 - MODELAGEM LINEAR PARA APRENDIZADO DE MÁQUINA (FIAP)
# Descrição: Implementação em Python de Regressão Linear Simples, Múltipla
#            e Regressão Logística Binária utilizando a base do FIFA 21.
# ==============================================================================

# Importação das bibliotecas necessárias para manipulação de dados, 
# visualização gráfica e modelagem estatística/machine learning.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seaborn_plt

# Bibliotecas de estatística avançada e machine learning
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Configuração para ignorar avisos visuais irrelevantes no terminal
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# CARREGAMENTO DA BASE DE DADOS
# ------------------------------------------------------------------------------
# Carrega o arquivo CSV do FIFA 21 utilizando o separador correto (ponto e vírgula)
caminho_arquivo = 'FIFA_21_Complete.csv'
df = pd.read_csv(caminho_arquivo, sep=';')

# Remove linhas com valores nulos para evitar inconsistências nos modelos
df = df.dropna()


# ==============================================================================
# 1. REGRESSÃO LINEAR SIMPLES
# ==============================================================================
print("="*60)
print("1. REGRESSÃO LINEAR SIMPLES")
print("="*60)

# Definindo a Variável Dependente (Y - Quantitativa Contínua) e Independente (X)
# Objetivo: Prever o 'overall' (habilidade geral) com base no 'potential' (potencial)
Y_simples_col = 'overall'
X_simples_col = 'potential'

X_simples = df[[X_simples_col]]
y_simples = df[Y_simples_col]

# PRÉ-REQUISITO 1: Avaliação da Linearidade via Gráfico de Dispersão
plt.figure(figsize=(6, 4))
seaborn_plt.scatterplot(x=df[X_simples_col], y=df[Y_simples_col], alpha=0.5)
plt.title('Dispersão: Potencial vs Overall (Linearidade)')
plt.xlabel('Potencial (potential)')
plt.ylabel('Habilidade Geral (overall)')
plt.show()

# CRIAÇÃO DO MODELO: Utilizando Statsmodels para obter o resumo estatístico completo
X_simples_sm = sm.add_constant(X_simples) # Adiciona o intercepto (beta_0)
modelo_simples = sm.OLS(y_simples, X_simples_sm).fit()

# ANÁLISE DE ADEQUAÇÃO DO MODELO SIMPLES (Exibe R², p-valores, coeficientes)
print(modelo_simples.summary())

# PRÉ-REQUISITO 2: Análise de Resíduos (Verificação de Normalidade)
residuos_simples = modelo_simples.resid
plt.figure(figsize=(6, 4))
seaborn_plt.histplot(residuos_simples, bins=30, kde=True)
plt.title('Histograma dos Resíduos - Simples (Normalidade)')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.show()


# ==============================================================================
# 2. REGRESSÃO LINEAR MÚLTIPLA
# ==============================================================================
print("\n" + "="*60)
print("2. REGRESSÃO LINEAR MÚLTIPLA")
print("="*60)

# Selecionando múltiplas variáveis explicativas (X) para prever o 'overall' (Y)
colunas_multiplas = ['potential', 'age', 'hits']
X_multipla = df[colunas_multiplas]
y_multipla = df[Y_simples_col]

# PRÉ-REQUISITO 1: Verificação de Multicolinearidade através do VIF (Variance Inflation Factor)
vif_data = pd.DataFrame()
vif_data["Variável"] = X_multipla.columns
vif_data["VIF"] = [variance_inflation_factor(X_multipla.values, i) for i in range(X_multipla.shape[1])]
print("Fatores de Inflação da Variância (VIF):")
print(vif_data)
print("\nNota: Valores de VIF abaixo de 5 indicam ausência de multicolinearidade severa.\n")

# CRIAÇÃO DO MODELO DE REGRESSÃO MÚLTIPLA
X_multipla_sm = sm.add_constant(X_multipla)
modelo_multipla = sm.OLS(y_multipla, X_multipla_sm).fit()

# ANÁLISE DE ADEQUAÇÃO DO MODELO MÚLTIPLO (Estatísticas F, t, R² Ajustado)
print(modelo_multipla.summary())


# ==============================================================================
# 3. REGRESSÃO LOGÍSTICA BINÁRIA
# ==============================================================================
print("\n" + "="*60)
print("3. REGRESSÃO LOGÍSTICA BINÁRIA")
print("="*60)

# Criando uma variável alvo dicotômica (Qualitativa Binária: 0 ou 1)
# Exemplo: Classificar se o jogador é considerado 'Jovem Promissor' (Idade <= 22 anos = 1, Senão = 0)
df['is_jovem_promissor'] = (df['age'] <= 22).astype(int)

# Definindo as variáveis preditoras e a variável alvo
colunas_log = ['potential', 'overall', 'hits']
target_log = 'is_jovem_promissor'

X_log = df[colunas_log]
y_log = df[target_log]

# Divisão dos dados em bases de Treino (70%) e Teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.3, random_state=42)

# PRÉ-REQUISITOS E CRIAÇÃO DO MODELO: Ajuste da Regresão Logística
modelo_logistico = LogisticRegression(max_iter=10000)
modelo_logistico.fit(X_train, y_train)

# Realizando predições na base de teste
y_pred = modelo_logistico.predict(X_test)
y_prob = modelo_logistico.predict_proba(X_test)[:, 1]

# ANÁLISE DE ADEQUAÇÃO DO MODELO LOGÍSTICO:
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred))

# Avaliação da performance global através da Área sob a Curva ROC (AUC-ROC)
auc_roc = roc_auc_score(y_test, y_prob)
print(f"Área sob a Curva ROC (AUC): {auc_roc:.4f}")

# ==============================================================================
# SIMULAÇÃO DE CENÁRIOS COM NOVOS DADOS (Regressão Logística)
# ==============================================================================
print("\n" + "-"*40)
print("SIMULAÇÃO DE CENÁRIOS")
print("-"*40)

# Criando um dado hipotético de jogador para testar o modelo treinado
# Ordem das colunas: ['potential', 'overall', 'hits']
novo_jogador = pd.DataFrame([[88.0, 84.0, 150.0]], columns=colunas_log)

predicao_cenario = modelo_logistico.predict(novo_jogador)
prob_cenario = modelo_logistico.predict_proba(novo_jogador)

resultado_texto = "Jovem Promissor (Sim)" if predicao_cenario[0] == 1 else "Não é Jovem Promissor (Não)"
print(f"Cenário Simulado - Classificação Prevista: {resultado_texto}")
print(f"Probabilidades [Classe 0 (Não), Classe 1 (Sim)]: {prob_cenario[0]}")