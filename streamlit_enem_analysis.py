# --- IMPORTAÇÕES ---
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAÇÕES STREAMLIT ---
st.set_page_config(layout="wide", page_title="Análise de Regressão ENEM")

st.title("📊 Análise de Regressão — Base ENEM Real")
st.caption("Arquivo carregado pelo usuário: **Enem_2024_Amostra_Perfeita (1).xlsx**")

# --- 1. CARREGAR A BASE REAL DO ENEM ---
uploaded_file = "Enem_2024_Amostra_Perfeita (1).xlsx"

df = pd.read_excel(uploaded_file)

st.subheader("📄 Visualização Inicial dos Dados")
st.dataframe(df.head())

# --- SELEÇÃO DE VARIÁVEIS PELO USUÁRIO ---
st.sidebar.header("Configurações do Modelo")

# escolha da variável Y
target_col = st.sidebar.selectbox(
    "Selecione a variável dependente (Y):",
    df.columns
)

# escolha das variáveis X
predictors = st.sidebar.multiselect(
    "Selecione as variáveis independentes (X):",
    df.columns
)

if len(predictors) == 0:
    st.warning("Selecione pelo menos uma variável independente.")
    st.stop()

Y = df[target_col]
X = df[predictors]
X_sm = sm.add_constant(X)

# --- 2. AJUSTAR O MODELO OLS ---
model = sm.OLS(Y, X_sm).fit()

st.header("2️⃣ Resumo do Modelo de Regressão")
st.code(model.summary().as_text())

# --- 3. MATRIZ DE CORRELAÇÃO ---
st.header("1️⃣ Matriz de Correlação")

corr = df[[target_col] + predictors].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# --- 4. VIF ---
def calculate_vif(X):
    Xc = sm.add_constant(X)
    vif = pd.DataFrame()
    vif["Variável"] = Xc.columns
    vif["VIF"] = [
        variance_inflation_factor(Xc.values, i)
        for i in range(Xc.shape[1])
    ]
    return vif[vif["Variável"] != "const"]

st.header("3️⃣ Multicolinearidade — VIF")
st.dataframe(calculate_vif(X))

# --- 5. RESÍDUOS / SUPOSIÇÕES ---
st.header("4️⃣ Diagnóstico das Suposições")

# linearidade e homocedasticidade
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.scatter(model.fittedvalues, model.resid)
ax2.axhline(0, color="red", linestyle="--")
ax2.set_xlabel("Valores previstos (ŷ)")
ax2.set_ylabel("Resíduos")
st.pyplot(fig2)

# normalidade
fig3, ax3 = plt.subplots(figsize=(6, 4))
sm.qqplot(model.resid, line="s", ax=ax3)
st.pyplot(fig3)

# --- 6. OUTLIERS ---
st.header("5️⃣ Outliers / Influência")

influence = model.get_influence()

# Cook's distance
cooks_d = influence.cooks_distance[0]
n = len(df)

fig4, ax4 = plt.subplots(figsize=(7, 4))
ax4.stem(np.arange(n), cooks_d, markerfmt=",")
ax4.axhline(4/n, color="red", linestyle="--")
st.pyplot(fig4)

# DFFITS
dffits = influence.dffits[0]
fig5, ax5 = plt.subplots(figsize=(7, 4))
ax5.stem(np.arange(n), dffits, markerfmt=",")
st.pyplot(fig5)

# --- 7. MÉTRICAS ---
st.header("6️⃣ Métricas do Modelo")

y_pred = model.predict(X_sm)
rmse = np.sqrt(mean_squared_error(Y, y_pred))
r2 = r2_score(Y, y_pred)

st.write(f"**RMSE:** {rmse:.4f}")
st.write(f"**R²:** {r2:.4f}")
st.write(f"**AIC:** {model.aic:.2f}")
st.write(f"**BIC:** {model.bic:.2f}")

# --- 8. ROC / AUC (SE FOR CLASSIFICAÇÃO) ---
st.header("7️⃣ Classificação (Opcional)")

cut = st.sidebar.checkbox("Transformar Y em classificação? (Y ≥ 800)")
if cut:
    Y_class = (Y >= 800).astype(int)
    log_model = LogisticRegression(solver="liblinear")
    log_model.fit(X, Y_class)
    prob = log_model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(Y_class, prob)
    auc = roc_auc_score(Y_class, prob)

    fig6, ax6 = plt.subplots(figsize=(7, 4))
    ax6.plot(fpr, tpr)
    ax6.plot([0, 1], [0, 1], "r--")
    st.pyplot(fig6)

    st.write(f"**AUC:** {auc:.4f}")
