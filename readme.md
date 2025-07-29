# 🧠 Predição de Salário com XGBoost

Este projeto aplica técnicas de Machine Learning com o algoritmo XGBoost para prever salários com base em dados de empregadores. O dataset foi obtido do Kaggle e o notebook realiza desde o carregamento dos dados até a avaliação do modelo.

---

## 📦 Estrutura do Projeto

- `dados-do-empregador-xgboost.ipynb`: Notebook com todo o processo de análise e modelagem.
- `dados-do-empregador-xgboost.py`: Versão script do notebook para execução em pipelines e reprodutibilidade.
- `README.md`: Este arquivo de explicação e documentação do repositório.

---

## 📊 Descrição do Dataset

- Fonte: Kaggle - [gmudit/employer-data](https://www.kaggle.com/datasets/gmudit/employer-data)
- Arquivo principal: `Employers_data.csv`
- Atributos:
  - `Education`, `JobTitle`, `YearsExperience`, `City`, `Gender`, `Age`, `Salary` (target), entre outros.

---

## 🚀 Etapas do Projeto

1. **Importação e download do dataset via `kagglehub`**
2. **Exploração de dados (EDA)**
   - Análise de nulos
   - Histogramas de salário
   - Matriz de correlação
3. **Pré-processamento**
   - Encoding de variáveis categóricas
   - Normalização de atributos numéricos
   - Separação entre treino e teste
4. **Modelagem com XGBoost**
   - Treinamento do modelo
   - Avaliação com métricas: RMSE, R²
   - Importância das features
5. **Visualizações**
   - Actual_Salary e Predicted_Salary
   - Importância das variáveis
