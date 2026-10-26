# FIAP - Checkpoint 6 | Modelagem Linear para Aprendizado de Máquina (MLAM)

Este repositório contém a documentação e os códigos do **Checkpoint 6** da disciplina **Modelagem Linear para Aprendizado de Máquina (MLAM)**, ministrada no curso de **Bacharelado em Ciência da Computação (Turma 1CCPQ - FIAP)**.

O projeto desenvolve e compara modelos de aprendizado de máquina supervisionado (Regressão Linear Simples, Regressão Linear Múltipla e Regressão Logística Binária) utilizando a base de dados do futebol profissional **FIFA 21**.


## 👥 Integrantes do Grupo

* **Matheus Jorge Santana** — RM 574166

**Professor Responsável:** Me. Eng. Rodolfo Magliari de Paiva **Turma:** 1CCPQ — FIAP


## 📊 Base de Dados

A base utilizada foi o **FIFA 2021 Complete Player Dataset** (Kaggle), contendo registros de **17.981 jogadores** e suas características físicas e técnicas.

### Atributos Principais:

* : Código de identificação do atleta.
* : Nome do jogador.
* : Nacionalidade.
* : Posição em campo.
* : Nota de habilidade geral atual (0 a 100) — *Variável Alvo das Regressões Lineares*.
* : Potencial de desenvolvimento futuro (0 a 100).
* : Idade do jogador.
* : Engajamento/popularidade do atleta.
* : Clube do atleta.


## 🔬 Modelos e Resultados

### 1\. Regressão Linear Simples

* **Objetivo:** Prever a partir de .
* **Equação Estimada (OLS):**
* **Pré-requisitos:** Análise de linearidade (scatter plot) e normalidade dos resíduos (histograma com KDE).
* **Métrica:** R² = 0.408 (40,8% de explicabilidade, p &lt; 0.001).

### 2\. Regressão Linear Múltipla

* **Objetivo:** Prever com múltiplas variáveis (, , ).
* **Equação Estimada (OLS):**
* **Pré-requisito:** Avaliação de multicolinearidade pelo VIF (: 1.43, : 1.17, : 1.24 — todos &lt; 5.0).
* **Métrica:** R² Ajustado = 0.804 (80,4% de explicabilidade, p &lt; 0.001).

### 3\. Regressão Logística Binária &amp; Simulação

* **Objetivo:** Classificar se o jogador é "Jovem Promissor" ().
* **Variável Alvo:** (1 = Sim, 0 = Não).
* **Variáveis Explicativas:** , , .
* **Desempenho:** Acurácia = 93%, AUC-ROC = 0.9717.
* **Simulação de Cenário:** Atleta hipotético com Potential = 88.0, Overall = 84.0 e Hits = 150.0 foi classificado como Não Jovem Promissor (probabilidade de 46,7% para classe 1 e 53,3% para classe 0).


## 🛠️ Tecnologias e Bibliotecas

* **Python 3.x**
* **Pandas &amp; NumPy** (manipulação de dados)
* **Matplotlib &amp; Seaborn** (visualizações gráficas)
* **Statsmodels** (modelagem OLS e VIF)
* **Scikit-Learn** (regressão logística, métricas e avaliação ROC)


## 📚 Bibliografia (Normas ABNT)

* MISHRA, Aayush. **FIFA 2021 Complete Player Dataset**. Kaggle, 2020.
* PAIVA, Rodolfo Magliari de. **Modelagem Linear para Aprendizado de Máquina: Checkpoint 6**. São Paulo: FIAP, 2026.
* HASTIE, Trevor; TIBSHIRANI, Robert; FRIEDMAN, Jerome. **The Elements of Statistical Learning: Data Mining, Inference, and Prediction**. 2\. ed. New York: Springer, 2009.
* MONTGOMERY, Douglas C.; PECK, Elizabeth A.; VINING, G. Geoffrey. **Introduction to Linear Regression Analysis**. 5\. ed. Hoboken: John Wiley &amp; Sons, 2012.

---

_Desenvolvido pela turma 1CCPQ — Bacharelado em Ciência da Computação (FIAP)._
