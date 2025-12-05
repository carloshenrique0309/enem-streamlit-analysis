# streamlit_enem_analysis.py
# Versão final: sem scipy, sem statsmodels — compatível com Streamlit Cloud FREE
import streamlit as st
import pandas as pd
import numpy as np
from math import log
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score,
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Análise ENEM (compatível com Streamlit Cloud Free)")
st.title("📊 Análise Estatística e Modelagem — ENEM (Sem SciPy / Sem Statsmodels)")
st.markdown("App adaptado para ambientes onde `scipy`/`statsmodels` não estão disponíveis. "
            "Exibe estatísticas, diagnósticos e regras práticas de interpretação.")

# ------------------ Auxiliares (numpy implementations) ------------------

def fit_ols_numpy(X, y, add_intercept=True):
    if add_intercept:
        X_design = np.column_stack([np.ones(len(X)), np.asarray(X)])
    else:
        X_design = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    n, p = X_design.shape
    XtX = X_design.T.dot(X_design)
    inv_XtX = np.linalg.pinv(XtX)
    beta = inv_XtX.dot(X_design.T).dot(y)
    y_hat = X_design.dot(beta).flatten()
    resid = (y.flatten() - y_hat)
    RSS = float((resid**2).sum())
    df_resid = max(n - p, 1)
    sigma2 = RSS / df_resid
    cov_beta = sigma2 * inv_XtX
    se_beta = np.sqrt(np.diag(cov_beta))
    t_stats = (beta.flatten() / se_beta)
    # R2
    y_mean = y.mean()
    TSS = float(((y - y_mean)**2).sum())
    R2 = 1 - RSS / TSS if TSS > 0 else np.nan
    # AIC/BIC (Gaussian likelihood)
    aic = n * np.log(RSS / n) + 2 * p
    bic = n * np.log(RSS / n) + p * np.log(n)
    return {
        "beta": beta.flatten(),
        "se_beta": se_beta,
        "t": t_stats,
        "y_hat": y_hat,
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

def compute_vif(dfX):
    Xmat = np.asarray(dfX)
    n, k = Xmat.shape
    vifs = {}
    for j in range(k):
        y = Xmat[:, j]
        X_others = np.delete(Xmat, j, axis=1)
        if X_others.shape[1] == 0:
            vifs[dfX.columns[j]] = np.nan
            continue
        lr = LinearRegression().fit(X_others, y)
        R2j = lr.score(X_others, y)
        vif = 1.0 / (1.0 - R2j) if (1.0 - R2j) != 0 else np.inf
        vifs[dfX.columns[j]] = vif
    return pd.DataFrame.from_dict(vifs, orient='index', columns=['VIF'])

def durbin_watson(resid):
    r = np.asarray(resid)
    d = np.sum(np.diff(r)**2) / np.sum(r**2)
    return float(d)

def breusch_pagan_statistic(resid, X):
    # LM = n * R2_from_regression_of_resid2_on_X
    y_bp = resid**2
    # regress y_bp on X with intercept
    X_design = np.column_stack([np.ones(len(X)), np.asarray(X)])
    beta = np.linalg.pinv(X_design.T.dot(X_design)).dot(X_design.T).dot(y_bp)
    yhat = X_design.dot(beta)
    SSR = ((yhat - y_bp.mean())**2).sum()
    SST = ((y_bp - y_bp.mean())**2).sum()
    R2 = SSR / SST if SST > 0 else 0.0
    LM = len(resid) * R2
    # Return LM and R2. For interpreting p-value: for df = k (number of regressors)
    return {"LM": float(LM), "R2": float(R2)}

def jarque_bera_stat(resid):
    r = np.asarray(resid)
    n = len(r)
    m2 = np.mean((r - r.mean())**2)
    m3 = np.mean((r - r.mean())**3)
    m4 = np.mean((r - r.mean())**4)
    skew = m3 / (m2**1.5) if m2 > 0 else 0.0
    kurt = m4 / (m2**2) if m2 > 0 else 0.0
    jb = n/6.0 * (skew**2 + (kurt - 3.0)**2 / 4.0)
    return {"JB": float(jb), "skew": float(skew), "kurtosis": float(kurt)}

def cooks_dffits_dfbetas_from_fit(fit):
    X = fit["X_design"]  # n x p
    invXtX = fit["inv_XtX"]  # p x p
    resid = fit["resid"]
    MSE = fit["sigma2"]
    n, p = X.shape
    # leverage
    X_inv = X.dot(invXtX)
    h = np.sum(X_inv * X, axis=1)
    # Cook's distance
    cooks = (resid**2) / (p * MSE) * (h / (1 - h)**2)
    # DFFITS
    with np.errstate(divide='ignore', invalid='ignore'):
        dffits = (resid / np.sqrt(MSE * (1 - h))) * np.sqrt(h)
    # DFBETAS
    invXtX_Xt = invXtX.dot(X.T)  # p x n
    denom = (1 - h)
    denom_safe = np.where(denom == 0, 1e-12, denom)
    delta_b = - (invXtX_Xt * resid) / denom_safe  # p x n
    se_beta = np.sqrt(np.diag(invXtX) * MSE).reshape(-1, 1)  # p x 1
    dfbetas = (delta_b / se_beta).T  # n x p
    return {"leverage": h, "cooks": cooks, "dffits": dffits, "dfbetas": dfbetas}

def pearson_r_and_perm_pvalues(df, n_perm=500):
    cols = df.columns
    n = len(df)
    rmat = df.corr()
    pmat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    # permutation p-values is expensive; we compute only upper triangle
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            x = df.iloc[:, i].dropna().values
            y = df.iloc[:, j].dropna().values
            # align by index not trivial here; assume no NaNs inside numeric cols after earlier dropna
            if len(x) < 4 or len(y) < 4:
                p = np.nan
            else:
                r_obs = np.corrcoef(x, y)[0,1]
                # permutation test
                count = 0
                for _ in range(n_perm):
                    yperm = np.random.permutation(y)
                    r_perm = np.corrcoef(x, yperm)[0,1]
                    if abs(r_perm) >= abs(r_obs):
                        count += 1
                p = (count + 1) / (n_perm + 1)
            pmat.iloc[i,j] = p
            pmat.iloc[j,i] = p
    return rmat, pmat

# ------------------ UI: load data ------------------
st.sidebar.header("Dados")
use_upload = st.sidebar.checkbox("Fazer upload do arquivo (.xlsx)", value=False)
if use_upload:
    uploaded = st.sidebar.file_uploader("Carregue o Excel (.xlsx)", type=["xlsx"])
    if uploaded is None:
        st.info("Faça upload do arquivo ou desmarque o upload para usar o arquivo padrão.")
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

st.write(f"Planilha: **{sheet}** — dimensão: {df.shape[0]} x {df.shape[1]}")
st.dataframe(df.head())

# numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("Nenhuma coluna numérica encontrada.")
    st.stop()

st.sidebar.header("Variáveis")
target = st.sidebar.selectbox("Escolha target (Y)", numeric_cols, index=0)
predictors = st.sidebar.multiselect("Escolha preditores (X) — vazio = todas as numéricas exceto Y", [c for c in numeric_cols if c != target])
if not predictors:
    predictors = [c for c in numeric_cols if c != target]

st.write("Target:", target)
st.write("Preditores:", predictors)

# drop NA rows
df_mod = df[[target] + predictors].dropna()
Y = df_mod[target].values
X_df = df_mod[predictors].copy()

# 1) correlation
st.header("1️⃣ Análise de Correlação")
corr = df_mod.corr()
st.subheader("Matriz de correlação (Pearson)")
st.dataframe(corr.style.background_gradient(cmap="coolwarm").format(precision=3))

st.subheader("P-valores (permutation test, 500 perm) — pode demorar dependendo do tamanho")
with st.spinner("Calculando p-values por permutação (500 permutações — paramétrico alternativa sem SciPy)..."):
    rmat, pmat = pearson_r_and_perm_pvalues(df_mod, n_perm=500)
st.subheader("Matriz de p-values (permutation)")
st.dataframe(pmat.style.format(precision=4).applymap(lambda v: 'background-color: #ffcccc' if (isinstance(v, float) and v < 0.05) else ''))

st.subheader("Dispersão (escolha preditor)")
scat = st.selectbox("Escolha preditor para scatter", predictors)
fig, ax = plt.subplots(figsize=(6,4))
sns.regplot(x=df_mod[scat], y=df_mod[target], scatter_kws={"alpha":0.4}, line_kws={"color":"red"}, ax=ax)
ax.set_xlabel(scat); ax.set_ylabel(target)
st.pyplot(fig)

# 2) variable selection
st.header("2️⃣ Seleção de Variáveis (forward / backward / stepwise)")
method = st.selectbox("Método", ["backward", "forward", "stepwise"])
alpha_in = st.number_input("p-valor para entrar (forward, threshold via |t|~2)", value=0.05, format="%.4f")
alpha_out = st.number_input("p-valor para sair (backward, threshold via |t|~2)", value=0.05, format="%.4f")

def stepwise_selection(Xdf, y, method="both", verbose=True):
    cols = list(Xdf.columns)
    included = []
    while True:
        changed = False
        if method in ("forward", "both"):
            excluded = [c for c in cols if c not in included]
            best_p = 1.0; best_col = None
            for c in excluded:
                cols_try = included + [c]
                fit = fit_ols_numpy(Xdf[cols_try].values, y, add_intercept=True)
                tvals = fit["t"]
                # take last coef t as proxy
                t_last = tvals[-1]
                # approximate p by |t|>2 rule for large df
                p_approx = 0.05 if abs(t_last) >= 2 else 0.32
                if p_approx < best_p:
                    best_p = p_approx; best_col = c
            if best_col and best_p < alpha_in:
                included.append(best_col); changed=True
                if verbose: st.write(f"Add {best_col} (approx p {best_p})")
        if method in ("backward", "both"):
            if included:
                fit = fit_ols_numpy(Xdf[included].values, y, add_intercept=True)
                tvals = fit["t"][1:]  # exclude intercept
                p_approx_series = [0.05 if abs(t)>=2 else 0.32 for t in tvals]
                worst_idx = np.argmax(p_approx_series)
                if p_approx_series[worst_idx] > alpha_out:
                    rem = included[worst_idx]
                    included.remove(rem); changed=True
                    if verbose: st.write(f"Drop {rem} (approx p {p_approx_series[worst_idx]})")
        if not changed:
            break
    return included

with st.spinner("Executando seleção (aproximação sem SciPy)..."):
    selected = stepwise_selection(X_df, Y, method=method, verbose=True)
st.success(f"Variáveis selecionadas: {selected}")

# Fit final
st.header("Modelo Final (OLS usando numpy linear algebra)")
X_selected = X_df[selected] if len(selected)>0 else X_df.copy()
fit = fit_ols_numpy(X_selected.values, Y, add_intercept=True)
coef_names = ["Intercept"] + list(X_selected.columns)
coef_table = pd.DataFrame({
    "coef": fit["beta"],
    "se": fit["se_beta"],
    "t-stat": fit["t"]
}, index=coef_names)
st.subheader("Coeficientes (estimativa, se, t-stat)")
st.dataframe(coef_table.style.format("{:.6g}"))

# Model summary
n = fit["n"]; p = fit["p"]
R2 = fit["R2"]; RSS = fit["RSS"]
RMSE = np.sqrt(RSS / n)
# F-statistic
SSR = fit["TSS"] - RSS
df_model = p - 1
df_resid = n - p
MSR = SSR / df_model if df_model>0 else np.nan
MSE = RSS / df_resid if df_resid>0 else np.nan
F_stat = MSR / MSE if MSE>0 else np.nan
st.write(f"R² = {R2:.6f}  |  RMSE = {RMSE:.6f}")
st.write(f"F-statistic = {F_stat:.6g}  (df_model={df_model}, df_resid={df_resid})")
st.write(f"AIC = {fit['aic']:.6g}  |  BIC = {fit['bic']:.6g}")

st.markdown("---")

# 3) diagnostics
st.header("3️⃣ Diagnóstico das Suposições")
# Residuals vs fitted
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(fit["y_hat"], fit["resid"], alpha=0.4)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Fitted"); ax.set_ylabel("Residuals")
st.pyplot(fig)
# Durbin-Watson
dw = durbin_watson(fit["resid"])
st.write(f"Durbin-Watson = {dw:.4f}  (≈2 sem autocorrelação)")

# Breusch-Pagan statistic (LM)
bp = breusch_pagan_statistic(fit["resid"], X_selected.values)
st.write(f"Breusch-Pagan LM = {bp['LM']:.6g}  (interpret: compare to chi2_crit(df=k) e.g. chi2_0.95(k) )")
st.write(f"R² (aux) = {bp['R2']:.6g}")

# Normality: Jarque-Bera
jb = jarque_bera_stat(fit["resid"])
st.write(f"Jarque-Bera statistic = {jb['JB']:.6g}; skew = {jb['skew']:.6g}; kurtosis = {jb['kurtosis']:.6g}")
st.info("Interpretação rápida: JB > 5.99 ⇒ rejeita normalidade ao nível 5% (df=2).")

# VIF
st.subheader("VIF (multicolinearidade)")
vif_df = compute_vif(X_selected)
st.dataframe(vif_df.style.format("{:.4f}"))

st.markdown("---")

# 4) Outliers
st.header("4️⃣ Outliers e Observações Influentes")
inf = cooks_dffits_dfbetas_from_fit(fit)
cooks = inf["cooks"]
dffits = inf["dffits"]
dfbetas = inf["dfbetas"]
h = inf["leverage"]

cooks_df = pd.DataFrame({"index": df_mod.index, "cooks": cooks, "dffits": dffits, "leverage": h}).set_index("index").sort_values("cooks", ascending=False)
st.subheader("Top 10 — Cook's distance")
st.dataframe(cooks_df.head(10).style.format("{:.6g}"))

st.subheader("Top 10 — |DFFITS|")
st.dataframe(pd.DataFrame({"index": df_mod.index, "abs_dffits": np.abs(dffits)}).set_index("index").sort_values("abs_dffits", ascending=False).head(10).style.format("{:.6g}"))

dfbetas_df = pd.DataFrame(dfbetas, index=df_mod.index, columns=coef_names)
st.subheader("DFBETAS (máximo absoluto por coeficiente)")
st.dataframe(dfbetas_df.abs().max().sort_values(ascending=False).to_frame("max_abs_dfbeta").style.format("{:.6g}"))

fig, ax = plt.subplots(figsize=(8,3))
ax.stem(np.arange(len(cooks)), cooks, markerfmt=",", basefmt=" ")
ax.set_xlabel("Observação"); ax.set_ylabel("Cook's distance")
st.pyplot(fig)

st.markdown("---")

# 5) Model metrics
st.header("5️⃣ Métricas do Modelo")
st.write(f"Observações: {n}  |  Parâmetros (p): {p}")
st.write(f"R²: {R2:.6f}  |  RMSE: {RMSE:.6f}")
st.write(f"AIC: {fit['aic']:.6g}  |  BIC: {fit['bic']:.6g}")
st.write(f"F-statistic: {F_stat:.6g}  (sem p-value exato aqui — se |t| > ~2, coef provavelmente significativo)")

st.markdown("---")

# 6) Comparison & pipeline
st.header("6️⃣ Comparação de Modelos e Validação Cruzada")
k = st.number_input("K (CV folds)", min_value=2, max_value=10, value=5)
kf = KFold(n_splits=k, shuffle=True, random_state=42)
lr = LinearRegression()

X_all = X_df.values
scores_all = -cross_val_score(lr, X_all, Y, cv=kf, scoring='neg_root_mean_squared_error')
scores_sel = -cross_val_score(lr, X_selected.values, Y, cv=kf, scoring='neg_root_mean_squared_error')

st.write("RMSE (CV) — All predictors: mean={:.4f}, std={:.4f}".format(scores_all.mean(), scores_all.std()))
st.write("RMSE (CV) — Selected predictors: mean={:.4f}, std={:.4f}".format(scores_sel.mean(), scores_sel.std()))
st.write("AIC/BIC (modelo selecionado): AIC={:.4f}, BIC={:.4f}".format(fit["aic"], fit["bic"]))

st.subheader("Classificação (opcional)")
do_class = st.checkbox("Transformar target em binário e treinar LogisticRegression", value=False)
if do_class:
    threshold = st.number_input("Threshold para classe positiva (Y >= )", value=int(np.nanmedian(Y)))
    y_bin = (Y >= threshold).astype(int)
    log = LogisticRegression(solver='liblinear', max_iter=1000)
    Xc = X_selected.values
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
    fpr, tpr, _ = roc_curve(y_bin, y_proba)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_bin, y_proba):.4f}")
    ax.plot([0,1],[0,1],'r--')
    ax.set_xlabel("1 - Especificidade (FPR)"); ax.set_ylabel("Sensibilidade (TPR)")
    ax.legend(); st.pyplot(fig)

st.markdown("---")
st.write("Análise concluída. Observação: onde foi necessário usar funções especializadas (p-values exatas para distribuições), o app exibe estatísticas e regras práticas de interpretação para compatibilidade com ambientes sem SciPy/statsmodels.")
