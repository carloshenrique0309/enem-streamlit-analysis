# streamlit_enem_analysis.py
# App reescrito para NÃO usar statsmodels (compatível com Streamlit Cloud FREE)
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2, f as f_dist
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score,
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(layout="wide", page_title="Análise ENEM (scikit-learn)")
st.title("📊 Análise Estatística e Modelagem — ENEM (sem statsmodels)")
st.markdown("App reescrito para rodar em Streamlit Cloud FREE (sem `statsmodels`).")

#
# --------------------------- Helpers: OLS via numpy ---------------------------
#
def fit_ols_numpy(X, y, add_intercept=True):
    """Fit OLS using numpy linear algebra. Returns dict with coef, se, t, p, predictions, resid, RSS, sigma2, cov_beta, inv_XtX."""
    if add_intercept:
        X_design = np.column_stack([np.ones(len(X)), np.asarray(X)])
    else:
        X_design = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    n, p = X_design.shape
    XtX = X_design.T.dot(X_design)
    inv_XtX = np.linalg.pinv(XtX)  # pseudo-inverse for stability
    beta = inv_XtX.dot(X_design.T).dot(y)  # p x 1
    y_hat = X_design.dot(beta)
    resid = (y - y_hat).flatten()
    RSS = float((resid**2).sum())
    df_resid = max(n - p, 1)
    sigma2 = RSS / df_resid
    cov_beta = sigma2 * inv_XtX
    se_beta = np.sqrt(np.diag(cov_beta)).reshape(-1, 1)
    t_stats = (beta / se_beta).flatten()
    # two-sided p-values from t-distribution
    from scipy.stats import t as t_dist
    p_values = (2 * (1 - t_dist.cdf(np.abs(t_stats), df=df_resid))).flatten()
    # R2
    y_mean = y.mean()
    TSS = float(((y - y_mean)**2).sum())
    R2 = 1 - RSS / TSS if TSS > 0 else np.nan
    # AIC/BIC (using Gaussian log-likelihood)
    aic = n * np.log(RSS / n) + 2 * p
    bic = n * np.log(RSS / n) + p * np.log(n)
    return {
        "beta": beta.flatten(),
        "se_beta": se_beta.flatten(),
        "t": t_stats,
        "p": p_values,
        "y_hat": y_hat.flatten(),
        "resid": resid,
        "RSS": RSS,
        "sigma2": sigma2,
        "cov_beta": cov_beta,
        "inv_XtX": inv_XtX,
        "n": n,
        "p": p,
        "R2": R2,
        "TSS": TSS,
        "aic": aic,
        "bic": bic,
        "X_design": X_design
    }

def compute_vif(X):
    """Compute VIF for a DataFrame X (no intercept)."""
    Xmat = np.asarray(X)
    n, k = Xmat.shape
    vifs = {}
    for j in range(k):
        y_j = Xmat[:, j]
        X_others = np.delete(Xmat, j, axis=1)
        if X_others.shape[1] == 0:
            vifs[X.columns[j]] = np.nan
            continue
        lr = LinearRegression().fit(X_others, y_j)
        R2_j = lr.score(X_others, y_j)
        vif = 1.0 / (1.0 - R2_j) if (1 - R2_j) != 0 else np.inf
        vifs[X.columns[j]] = vif
    return pd.DataFrame.from_dict(vifs, orient='index', columns=['VIF'])

def durbin_watson(resid):
    """Durbin-Watson statistic."""
    resid = np.asarray(resid)
    diff = np.diff(resid)
    return float(np.sum(diff**2) / np.sum(resid**2))

def breusch_pagan_test(resid, X):
    """Breusch-Pagan test using regression of resid^2 on X (with intercept)."""
    # regress resid^2 on X (including intercept)
    y_bp = resid**2
    ols = fit_ols_numpy(X, y_bp, add_intercept=True)
    R2_bp = ols["R2"]
    n = ols["n"]
    LM = n * R2_bp
    df = X.shape[1]  # excluding intercept because X includes only exogenous vars
    pval = 1 - chi2.cdf(LM, df)
    return {"LM": LM, "pvalue": pval, "R2": R2_bp}

def cooks_distance_all(fit):
    """Compute leverage h, Cook's distance, DFFITS, DFBETAS for all observations given fit dict from fit_ols_numpy."""
    X = fit["X_design"]  # n x p
    invXtX = fit["inv_XtX"]  # p x p
    resid = fit["resid"]  # n
    sigma2 = fit["sigma2"] if "sigma2" in fit else fit["sigma2"] if "sigma2" in fit else fit["sigma2"]
    # but fit uses 'sigma2' key name? it's 'sigma2' in returned dict? we used 'sigma2'
    MSE = fit["sigma2"]
    n, p = X.shape
    # leverage h_ii = row_i * invXtX * row_i^T
    X_inv = X.dot(invXtX)  # n x p
    h_ii = np.sum(X_inv * X, axis=1)
    # Cook's distance
    cooks = (resid**2) / (p * MSE) * (h_ii / (1 - h_ii)**2)
    # DFFITS
    dffits = (resid / (np.sqrt(MSE * (1 - h_ii)))) * np.sqrt(h_ii)
    # DFBETAS: delta_beta_i = - invXtX @ x_i * resid_i / (1 - h_ii)
    # compute invXtX @ X.T => p x n
    invXtX_Xt = invXtX.dot(X.T)  # p x n
    denom = (1 - h_ii)
    denom_safe = np.where(denom == 0, 1e-12, denom)
    # delta_betas: p x n matrix
    delta_betas = - (invXtX_Xt * resid) / denom_safe  # broadcasting (p x n)
    # se_beta_j = sqrt(MSE * invXtX[j,j])
    se_beta = np.sqrt(np.diag(invXtX) * MSE).reshape(-1, 1)  # p x 1
    # dfbetas: n x p
    dfbetas = (delta_betas / se_beta).T  # n x p
    return {
        "h_ii": h_ii,
        "cooks": cooks,
        "dffits": dffits,
        "dfbetas": dfbetas
    }

#
# --------------------------- UI: load data -----------------------------------
#
st.sidebar.header("Dados")
use_upload = st.sidebar.checkbox("Fazer upload do arquivo (se quiser)", value=False)
if use_upload:
    uploaded = st.sidebar.file_uploader("Carregue o Excel (.xlsx)", type=["xlsx"])
    if uploaded is None:
        st.info("Faça upload do arquivo ou desmarque a opção para usar o arquivo do repositório.")
        st.stop()
    xls = pd.ExcelFile(uploaded)
    sheet = st.sidebar.selectbox("Escolha a planilha", xls.sheet_names)
    df = pd.read_excel(uploaded, sheet_name=sheet, engine="openpyxl")
else:
    DEFAULT_PATH = "/mnt/data/Enem_2024_Amostra_Perfeita (1).xlsx"
    try:
        xls = pd.ExcelFile(DEFAULT_PATH)
        sheet = st.sidebar.selectbox("Escolha a planilha", xls.sheet_names)
        df = pd.read_excel(DEFAULT_PATH, sheet_name=sheet, engine="openpyxl")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo padrão: {e}")
        st.stop()

st.write(f"Planilha selecionada: **{sheet}** — dimensão: {df.shape[0]} x {df.shape[1]}")
st.dataframe(df.head())

# Only numeric columns as candidate predictors/targets
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("Não foram encontradas colunas numéricas no arquivo.")
    st.stop()

st.sidebar.header("Variáveis")
target = st.sidebar.selectbox("Escolha a variável alvo (Y)", numeric_cols, index=0)
predictors = st.sidebar.multiselect("Escolha preditores (se vazio, usa todas as numéricas exceto Y)", [c for c in numeric_cols if c != target])
if not predictors:
    predictors = [c for c in numeric_cols if c != target]

st.write("**Target (Y):**", target)
st.write("**Preditores (X):**", predictors)

# Drop rows with NA in chosen columns
df_mod = df[[target] + predictors].dropna()
Y = df_mod[target].values
X_df = df_mod[predictors].copy()

#
# --------------------------- 1) Correlação -----------------------------------
#
st.header("1️⃣ Análise de Correlação")
corr = df_mod.corr()
st.subheader("Matriz de Correlação (Pearson)")
st.dataframe(corr.style.background_gradient(cmap="coolwarm").format(precision=3))

# compute p-values for Pearson
def pearson_pvalues(df_numeric):
    cols = df_numeric.columns
    pmat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            xi = df_numeric.iloc[:, i]
            xj = df_numeric.iloc[:, j]
            common = xi.index.intersection(xj.index)
            if len(common) > 2:
                r, p = stats.pearsonr(xi.loc[common], xj.loc[common])
            else:
                r, p = np.nan, np.nan
            pmat.iloc[i, j] = p
            pmat.iloc[j, i] = p
    return pmat

pvals = pearson_pvalues(df_mod)
st.subheader("P-valores das correlações (Pearson)")
st.dataframe(pvals.style.format(precision=4).applymap(lambda v: 'background-color: #ffcccc' if (isinstance(v, float) and v < 0.05) else ''))

st.subheader("Gráfico de dispersão (escolha preditor)")
scat = st.selectbox("Escolha o preditor para scatter", predictors)
fig, ax = plt.subplots(figsize=(6,4))
sns.regplot(x=df_mod[scat], y=df_mod[target], scatter_kws={"alpha":0.4}, line_kws={"color":"red"}, ax=ax)
ax.set_xlabel(scat)
ax.set_ylabel(target)
st.pyplot(fig)

#
# --------------------------- 2) Seleção de variáveis -------------------------
#
st.header("2️⃣ Seleção de Variáveis (forward / backward / stepwise)")
method = st.selectbox("Método", ["backward", "forward", "stepwise"])
alpha_in = st.number_input("p-valor para entrar (forward)", value=0.05, format="%.4f")
alpha_out = st.number_input("p-valor para sair (backward)", value=0.05, format="%.4f")

def stepwise_selection_numpy(X_df, y, initial_list=[], threshold_in=0.05, threshold_out=0.05, verbose=False, direction="both"):
    """Perform stepwise selection using OLS p-values computed by fit_ols_numpy."""
    X_cols = list(X_df.columns)
    included = list(initial_list)
    while True:
        changed = False
        # forward
        if direction in ("forward", "both"):
            excluded = [c for c in X_cols if c not in included]
            best_pval = 1.0
            best_feature = None
            for new_col in excluded:
                cols = included + [new_col]
                fit = fit_ols_numpy(X_df[cols].values, y, add_intercept=True)
                # p-value of the last added coef (position = len(cols)) (intercept=0)
                pval = fit["p"][-1]
                if pval < best_pval:
                    best_pval = pval
                    best_feature = new_col
            if best_feature is not None and best_pval < threshold_in:
                included.append(best_feature)
                changed = True
                if verbose:
                    st.write(f"Add {best_feature} with p={best_pval:.6g}")
        # backward
        if direction in ("backward", "both"):
            if included:
                fit = fit_ols_numpy(X_df[included].values, y, add_intercept=True)
                pvals = pd.Series(fit["p"][1:], index=included)  # exclude intercept
                worst_pval = pvals.max()
                if worst_pval > threshold_out:
                    worst_feature = pvals.idxmax()
                    included.remove(worst_feature)
                    changed = True
                    if verbose:
                        st.write(f"Drop {worst_feature} with p={worst_pval:.6g}")
        if not changed:
            break
    return included

with st.spinner("Executando seleção..."):
    selected_vars = stepwise_selection_numpy(X_df, Y, initial_list=[], threshold_in=alpha_in, threshold_out=alpha_out, verbose=True, direction=method)
st.success(f"Variáveis selecionadas: {selected_vars}")

#
# --------------------------- Fit final OLS (numpy) --------------------------
#
st.header("Modelo Final (OLS via numpy)")

X_sel = X_df[selected_vars] if len(selected_vars) > 0 else X_df.copy()
fit = fit_ols_numpy(X_sel.values, Y, add_intercept=True)

coef_names = ["Intercept"] + selected_vars
coef_table = pd.DataFrame({
    "coef": fit["beta"],
    "se": fit["se_beta"],
    "t": fit["t"],
    "p-value": fit["p"]
}, index=coef_names)
st.subheader("Coeficientes (estimativa, se, t, p)")
st.dataframe(coef_table.style.format("{:.6g}"))

# Model summary numbers
y_hat = fit["y_hat"]
resid = fit["resid"]
n = fit["n"]
p = fit["p"]
R2 = fit["R2"]
RMSE = np.sqrt(fit["RSS"] / n)
st.write(f"R² = {R2:.6f}")
st.write(f"RMSE = {RMSE:.6f}")
# F-test
SSR = fit["TSS"] - fit["RSS"]
df_model = p - 1
df_resid = n - p
MSR = SSR / df_model if df_model > 0 else np.nan
MSE = fit["RSS"] / df_resid if df_resid > 0 else np.nan
F_stat = MSR / MSE if MSE > 0 else np.nan
F_p = 1 - f_dist.cdf(F_stat, df_model, df_resid) if not np.isnan(F_stat) else np.nan
st.write(f"F = {F_stat:.6f}, p-value = {F_p:.6g}")
st.write(f"AIC = {fit['aic']:.6f}, BIC = {fit['bic']:.6f}")

st.markdown("---")

#
# --------------------------- 3) Diagnóstico das suposições -------------------
#
st.header("3️⃣ Diagnóstico das Suposições")

# Residuals vs Fitted (linearity & homoscedasticity)
st.subheader("Linearidade e Homocedasticidade — Resíduos vs Fitted")
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(y_hat, resid, alpha=0.5)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Valores preditos")
ax.set_ylabel("Resíduos")
st.pyplot(fig)

# Durbin-Watson (independência)
dw = durbin_watson(resid)
st.write(f"Durbin-Watson = {dw:.4f} (≈2 sem autocorrelação)")

# Breusch-Pagan test
bp = breusch_pagan_test(resid, X_sel.values)
st.write("Breusch-Pagan (heteroscedasticity): LM = {:.4f}, p-value = {:.6g}".format(bp["LM"], bp["pvalue"]))

# Normalidade dos resíduos
st.subheader("Normalidade dos Erros")
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
stats.probplot(resid, dist="norm", plot=ax)
st.pyplot(fig)
# Shapiro (cuidado com n muito grande)
if len(resid) <= 5000:
    sh_stat, sh_p = stats.shapiro(resid)
    st.write(f"Shapiro-Wilk: stat={sh_stat:.6g}, p={sh_p:.6g}")
else:
    # D'Agostino K^2 test
    k2_stat, k2_p = stats.normaltest(resid)
    st.write(f"D'Agostino K^2: stat={k2_stat:.6g}, p={k2_p:.6g}")

# VIF
st.subheader("Multicolinearidade — VIF")
vif_df = compute_vif(X_sel)
st.dataframe(vif_df.style.format("{:.4f}"))

st.markdown("---")

#
# --------------------------- 4) Outliers & Influence ------------------------
#
st.header("4️⃣ Outliers e Observações Influentes")
inf = cooks_distance_all({
    "X_design": fit["X_design"],
    "inv_XtX": fit["inv_XtX"],
    "resid": fit["resid"],
    "sigma2": fit["sigma2"] if "sigma2" in fit else fit["sigma2"],
    "p": fit["p"],
    # compatibility
})
# Top cooks
cooks = inf["cooks"]
dffits = inf["dffits"]
dfbetas = inf["dfbetas"]  # n x p
h = inf["h_ii"]

cooks_df = pd.DataFrame({
    "index": df_mod.index,
    "cooks": cooks,
    "dffits": dffits,
    "leverage": h
}).set_index("index").sort_values("cooks", ascending=False)

st.subheader("Top 10 — Cook's distance")
st.dataframe(cooks_df.head(10).style.format("{:.6g}"))

st.subheader("Top 10 — |DFFITS|")
st.dataframe(pd.DataFrame({"index": df_mod.index, "abs_dffits": np.abs(dffits)}).set_index("index").sort_values("abs_dffits", ascending=False).head(10).style.format("{:.6g}"))

# DFBETAS: show max absolute per coefficient
dfbetas_df = pd.DataFrame(dfbetas, index=df_mod.index, columns=coef_names)
dfbetas_max = dfbetas_df.abs().max().sort_values(ascending=False)
st.subheader("DFBETAS (máximo absoluto por coeficiente)")
st.dataframe(dfbetas_max.to_frame("max_abs_dfbeta").style.format("{:.6g}"))

# Plot Cook's
fig, ax = plt.subplots(figsize=(8,3))
ax.stem(np.arange(len(cooks)), cooks, markerfmt=",", basefmt=" ")
ax.set_xlabel("Observação (ordem)")
ax.set_ylabel("Cook's distance")
st.pyplot(fig)

st.markdown("---")

#
# --------------------------- 5) Métricas do modelo --------------------------
#
st.header("5️⃣ Métricas do Modelo")
st.write(f"Observações usadas: {fit['n']}")
st.write(f"Parâmetros (p): {fit['p']}")
st.write(f"R²: {fit['R2']:.6f}")
st.write(f"RMSE: {np.sqrt(fit['RSS']/fit['n']):.6f}")
st.write(f"AIC: {fit['aic']:.6f}, BIC: {fit['bic']:.6f}")
st.write(f"F-statistic: {F_stat:.6f}, p-value: {F_p:.6g}")

st.markdown("---")

#
# --------------------------- 6) Comparação / Pipeline -----------------------
#
st.header("6️⃣ Comparação e Seleção de Modelos (K-Fold / Classificação)")

k = st.number_input("K para K-Fold (CV)", min_value=2, max_value=10, value=5)
kf = KFold(n_splits=k, shuffle=True, random_state=42)
lr = LinearRegression()

# CV RMSE on full X vs selected
X_all = X_df.values
Y_all = Y
scores_all = -cross_val_score(lr, X_all, Y_all, cv=kf, scoring='neg_root_mean_squared_error')
scores_sel = -cross_val_score(lr, X_sel.values, Y_all, cv=kf, scoring='neg_root_mean_squared_error')

st.write("RMSE (CV) — All predictors: mean={:.4f}, std={:.4f}".format(scores_all.mean(), scores_all.std()))
st.write("RMSE (CV) — Selected predictors: mean={:.4f}, std={:.4f}".format(scores_sel.mean(), scores_sel.std()))
st.write("AIC/BIC do modelo selecionado: AIC={:.4f}, BIC={:.4f}".format(fit["aic"], fit["bic"]))

# Classification optional
st.subheader("Métricas de Classificação (opcional)")
do_class = st.checkbox("Transformar target em binário e treinar LogisticRegression", value=False)
if do_class:
    threshold = st.number_input("Threshold para classe positiva (Y >= )", value=int(np.nanmedian(Y_all)))
    y_bin = (Y_all >= threshold).astype(int)
    log = LogisticRegression(solver='liblinear', max_iter=1000)
    Xc = X_sel.values
    log.fit(Xc, y_bin)
    y_pred = log.predict(Xc)
    y_proba = log.predict_proba(Xc)[:,1]
    cm = confusion_matrix(y_bin, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    st.write(f"Acurácia: {accuracy_score(y_bin, y_pred):.4f}")
    st.write(f"Precisão: {precision_score(y_bin, y_pred, zero_division=0):.4f}")
    st.write(f"Sensibilidade (Recall): {recall_score(y_bin, y_pred):.4f}")
    st.write(f"Especificidade: {specificity:.4f}")
    st.write(f"F1 Score: {f1_score(y_bin, y_pred):.4f}")
    st.write(f"AUC: {roc_auc_score(y_bin, y_proba):.4f}")
    # ROC plot
    fpr, tpr, _ = roc_curve(y_bin, y_proba)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_bin, y_proba):.4f}")
    ax.plot([0,1],[0,1], 'r--')
    ax.set_xlabel("1 - Especificidade (FPR)")
    ax.set_ylabel("Sensibilidade (TPR)")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.write("Análise concluída. Ajuste seleções na barra lateral se quiser repetir com outras variáveis.")
