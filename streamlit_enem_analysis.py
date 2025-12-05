# streamlit_enem_analysis.py
# Versão final SEM scikit-learn, SEM scipy, SEM statsmodels.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import log

st.set_page_config(layout="wide", page_title="Análise ENEM (sem sklearn/scipy/statsmodels)")
st.title("📊 Análise ENEM — Compatível com Streamlit Cloud FREE")
st.markdown("App que usa apenas numpy/pandas/matplotlib/seaborn/openpyxl/streamlit.")

# ----------------------- UTILIDADES NUMPY-BASED -----------------------

def ols_fit(X, y, add_intercept=True):
    """Fit OLS via normal equations. Returns dict with many useful quantities."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    if add_intercept:
        X_design = np.column_stack([np.ones(len(X)), X])
    else:
        X_design = X
    n, p = X_design.shape
    XtX = X_design.T @ X_design
    invXtX = np.linalg.pinv(XtX)
    beta = invXtX @ X_design.T @ y
    y_hat = (X_design @ beta).flatten()
    resid = (y.flatten() - y_hat)
    RSS = float((resid**2).sum())
    df_resid = max(n - p, 1)
    sigma2 = RSS / df_resid
    cov_beta = sigma2 * invXtX
    se_beta = np.sqrt(np.diag(cov_beta))
    t_stats = (beta.flatten() / se_beta)
    y_mean = y.mean()
    TSS = float(((y - y_mean)**2).sum())
    R2 = 1 - RSS / TSS if TSS > 0 else np.nan
    aic = n * np.log(RSS / n) + 2 * p
    bic = n * np.log(RSS / n) + p * np.log(n)
    return {
        "beta": beta.flatten(),
        "se": se_beta,
        "t": t_stats,
        "y_hat": y_hat,
        "resid": resid,
        "RSS": RSS,
        "sigma2": sigma2,
        "cov_beta": cov_beta,
        "invXtX": invXtX,
        "n": n,
        "p": p,
        "R2": R2,
        "TSS": TSS,
        "aic": aic,
        "bic": bic,
        "X_design": X_design
    }

def compute_vif(dfX):
    """VIF via OLS R2 of each variable regressed on the others (no intercept included inside)."""
    X = np.asarray(dfX)
    n, k = X.shape
    vifs = {}
    for j in range(k):
        y = X[:, j]
        X_others = np.delete(X, j, axis=1)
        if X_others.shape[1] == 0:
            vifs[dfX.columns[j]] = np.nan
            continue
        beta = np.linalg.pinv(X_others.T @ X_others) @ X_others.T @ y
        yhat = X_others @ beta
        ssr = ((yhat - y.mean())**2).sum()
        sst = ((y - y.mean())**2).sum()
        R2 = ssr / sst if sst > 0 else 0.0
        vif = 1.0 / (1.0 - R2) if (1.0 - R2) != 0 else np.inf
        vifs[dfX.columns[j]] = vif
    return pd.DataFrame.from_dict(vifs, orient='index', columns=['VIF'])

def durbin_watson(resid):
    r = np.asarray(resid)
    return float(np.sum(np.diff(r)**2) / np.sum(r**2))

def breusch_pagan_stat(resid, X):
    # regress resid^2 on X with intercept -> compute R2 and LM = n * R2
    y_bp = resid**2
    X_design = np.column_stack([np.ones(len(X)), np.asarray(X)])
    beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y_bp
    yhat = X_design @ beta
    ssr = ((yhat - y_bp.mean())**2).sum()
    sst = ((y_bp - y_bp.mean())**2).sum()
    R2 = ssr / sst if sst > 0 else 0.0
    LM = len(resid) * R2
    return {"LM": float(LM), "R2": float(R2)}

def jarque_bera(resid):
    r = np.asarray(resid)
    n = len(r)
    m2 = np.mean((r - r.mean())**2)
    m3 = np.mean((r - r.mean())**3)
    m4 = np.mean((r - r.mean())**4)
    skew = m3 / (m2**1.5) if m2>0 else 0.0
    kurt = m4 / (m2**2) if m2>0 else 0.0
    jb = n/6.0 * (skew**2 + (kurt - 3.0)**2 / 4.0)
    return {"JB": float(jb), "skew": float(skew), "kurtosis": float(kurt)}

def influence_measures(fit):
    X = fit["X_design"]
    invXtX = fit["invXtX"]
    resid = fit["resid"]
    MSE = fit["sigma2"]
    n, p = X.shape
    X_inv = X @ invXtX
    h = np.sum(X_inv * X, axis=1)
    denom = (1 - h)
    denom_safe = np.where(denom == 0, 1e-12, denom)
    cooks = (resid**2) / (p * MSE) * (h / (denom_safe**2))
    with np.errstate(divide='ignore', invalid='ignore'):
        dffits = (resid / np.sqrt(MSE * (1 - h))) * np.sqrt(h)
    invXtX_Xt = invXtX @ X.T
    delta_b = - (invXtX_Xt * resid) / denom_safe  # p x n
    se_beta = np.sqrt(np.diag(invXtX) * MSE).reshape(-1, 1)
    dfbetas = (delta_b / se_beta).T
    return {"leverage": h, "cooks": cooks, "dffits": dffits, "dfbetas": dfbetas}

# ----------------------- LOGISTIC (Newton-Raphson) -----------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_fit_newton(X, y, add_intercept=True, max_iter=50, tol=1e-6):
    """Fit logistic regression via Newton-Raphson. Returns dict with params and fitted probs."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    if add_intercept:
        X_design = np.column_stack([np.ones(len(X)), X])
    else:
        X_design = X
    n, p = X_design.shape
    beta = np.zeros((p, 1))
    for it in range(max_iter):
        z = X_design @ beta
        mu = sigmoid(z)
        W = (mu * (1 - mu)).flatten()
        # avoid zeros
        W_safe = np.where(W == 0, 1e-12, W)
        # gradient and Hessian
        grad = X_design.T @ (y - mu)
        # Hessian = - X^T W X
        XW = X_design * W_safe.reshape(-1, 1)
        H = -(X_design.T @ XW)
        try:
            delta = np.linalg.pinv(H) @ grad
        except Exception:
            delta = np.linalg.pinv(H + 1e-6 * np.eye(p)) @ grad
        beta_new = beta - delta
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    preds = sigmoid(X_design @ beta).flatten()
    return {"beta": beta.flatten(), "proba": preds, "X_design": X_design}

def roc_auc_from_probs(y_true, probs):
    """Compute ROC AUC (trapezoidal) from true binary labels and predicted probabilities."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    # sort by prob desc
    desc_idx = np.argsort(-probs)
    t = y_true[desc_idx]
    tp = np.cumsum(t)
    fp = np.cumsum(1 - t)
    tp_rate = tp / max(tp[-1], 1)
    fp_rate = fp / max(fp[-1], 1)
    # add (0,0) at start
    x = np.concatenate([[0.0], fp_rate])
    y = np.concatenate([[0.0], tp_rate])
    auc = 0.0
    for i in range(1, len(x)):
        auc += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0
    return abs(auc)

# ----------------------- UI / Fluxo -----------------------

st.sidebar.header("Dados")
use_upload = st.sidebar.checkbox("Fazer upload do arquivo (.xlsx)", value=False)
if use_upload:
    uploaded = st.sidebar.file_uploader("Excel (.xlsx)", type=["xlsx"])
    if uploaded is None:
        st.info("Faça upload do arquivo ou desmarque para usar arquivo padrão.")
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
        st.error(f"Erro ao ler arquivo padrão: {e}")
        st.stop()

st.write(f"Planilha: **{sheet}** — dimensão: {df.shape[0]} x {df.shape[1]}")
st.dataframe(df.head())

# numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("Não há colunas numéricas.")
    st.stop()

st.sidebar.header("Variáveis")
target = st.sidebar.selectbox("Escolha target (Y)", numeric_cols, index=0)
predictors = st.sidebar.multiselect("Escolha preditores (X) — vazio = todas as numéricas exceto Y", [c for c in numeric_cols if c != target])
if not predictors:
    predictors = [c for c in numeric_cols if c != target]

st.write("Target:", target)
st.write("Preditores:", predictors)

# prepare data
df_mod = df[[target] + predictors].dropna()
Y = df_mod[target].values
X_df = df_mod[predictors].copy()

# 1) CORRELAÇÃO
st.header("1️⃣ Análise de Correlação")
corr = df_mod.corr()
st.subheader("Matriz de Correlação (Pearson)")
st.dataframe(corr.style.background_gradient(cmap="coolwarm").format(precision=3))

# permutation p-values (expensive but safe)
def pearson_perm_pvalues(df_numeric, n_perm=300):
    cols = df_numeric.columns
    pmat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            x = df_numeric.iloc[:, i].values
            y = df_numeric.iloc[:, j].values
            r_obs = np.corrcoef(x, y)[0,1]
            count = 0
            for _ in range(n_perm):
                yperm = np.random.permutation(y)
                r_perm = np.corrcoef(x, yperm)[0,1]
                if abs(r_perm) >= abs(r_obs):
                    count += 1
            p = (count + 1) / (n_perm + 1)
            pmat.iloc[i,j] = p
            pmat.iloc[j,i] = p
    return pmat

st.subheader("P-valores por permutação (Pearson) — pode demorar")
with st.spinner("Calculando p-values por permutação..."):
    pvals = pearson_perm_pvalues(df_mod, n_perm=300)
st.dataframe(pvals.style.format(precision=4).applymap(lambda v: 'background-color: #ffcccc' if (isinstance(v,float) and v < 0.05) else ''))

st.subheader("Gráfico de dispersão")
scat = st.selectbox("Escolha preditor para scatter", predictors)
fig, ax = plt.subplots(figsize=(6,4))
sns.regplot(x=df_mod[scat], y=df_mod[target], scatter_kws={"alpha":0.4}, line_kws={"color":"red"}, ax=ax)
st.pyplot(fig)

# 2) SELEÇÃO DE VARIÁVEIS (aproximação usando t-stat ~ |t| > 2)
st.header("2️⃣ Seleção de Variáveis (forward / backward / stepwise)")
method = st.selectbox("Método", ["backward", "forward", "stepwise"])
alpha_in = st.number_input("p-valor p/ entrar (aprox)", value=0.05, format="%.4f")
alpha_out = st.number_input("p-valor p/ sair (aprox)", value=0.05, format="%.4f")

def stepwise(Xdf, y, method="both", verbose=True):
    cols = list(Xdf.columns)
    included = []
    while True:
        changed = False
        if method in ("forward","both"):
            excluded = [c for c in cols if c not in included]
            best_col = None; best_score = 1.0
            for c in excluded:
                cols_try = included + [c]
                fit = ols_fit(Xdf[cols_try].values, y, add_intercept=True)
                t_last = fit["t"][-1]
                p_approx = 0.05 if abs(t_last) >= 2 else 0.32
                if p_approx < best_score:
                    best_score = p_approx; best_col = c
            if best_col is not None and best_score < alpha_in:
                included.append(best_col); changed=True
                if verbose: st.write(f"Add {best_col} (approx p {best_score})")
        if method in ("backward","both"):
            if included:
                fit = ols_fit(Xdf[included].values, y, add_intercept=True)
                tvals = fit["t"][1:]  # exclude intercept
                p_approx = [0.05 if abs(t)>=2 else 0.32 for t in tvals]
                worst_idx = int(np.argmax(p_approx))
                if p_approx[worst_idx] > alpha_out:
                    rem = included[worst_idx]
                    included.remove(rem); changed=True
                    if verbose: st.write(f"Drop {rem} (approx p {p_approx[worst_idx]})")
        if not changed:
            break
    return included

with st.spinner("Selecionando variáveis..."):
    selected = stepwise(X_df, Y, method=method, verbose=True)
st.success(f"Selecionadas: {selected}")

# Fit final
st.header("Modelo Final — OLS (numpy)")
X_sel = X_df[selected] if len(selected)>0 else X_df.copy()
fit = ols_fit(X_sel.values, Y, add_intercept=True)
coef_names = ["Intercept"] + list(X_sel.columns)
coef_df = pd.DataFrame({
    "coef": fit["beta"],
    "se": fit["se"],
    "t-stat": fit["t"]
}, index=coef_names)
st.subheader("Coeficientes")
st.dataframe(coef_df.style.format("{:.6g}"))

st.write(f"R² = {fit['R2']:.6f}")
RMSE = np.sqrt(fit["RSS"]/fit["n"])
st.write(f"RMSE = {RMSE:.6f}")
SSR = fit["TSS"] - fit["RSS"]
df_model = fit["p"] - 1
df_resid = fit["n"] - fit["p"]
MSR = SSR / df_model if df_model>0 else np.nan
MSE = fit["RSS"] / df_resid if df_resid>0 else np.nan
F_stat = MSR / MSE if MSE>0 else np.nan
st.write(f"F-statistic = {F_stat:.6g} (df_model={df_model}, df_resid={df_resid})")
st.write(f"AIC = {fit['aic']:.6g}  BIC = {fit['bic']:.6g}")

# 3) DIAGNÓSTICO
st.header("3️⃣ Diagnóstico das Suposições")
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(fit["y_hat"], fit["resid"], alpha=0.4)
ax.axhline(0, color="red", linestyle="--")
st.pyplot(fig)

dw = durbin_watson(fit["resid"])
st.write(f"Durbin-Watson = {dw:.4f} (≈2 indica não-autocorrelação)")

bp = breusch_pagan_stat(fit["resid"], X_sel.values)
st.write(f"Breusch-Pagan LM = {bp['LM']:.6g}  (interpretar comparando com chi2 critério df = k)")

jb = jarque_bera(fit["resid"])
st.write(f"Jarque-Bera = {jb['JB']:.6g}; skew = {jb['skew']:.6g}; kurtosis = {jb['kurtosis']:.6g}")
st.info("Interpretação: JB > 5.99 ⇒ rejeita normalidade ao nível 5% (aprox).")

vif_df = compute_vif(X_sel)
st.subheader("VIF")
st.dataframe(vif_df.style.format("{:.4f}"))

# 4) OUTLIERS & INFLUENCE
st.header("4️⃣ Outliers e Observações Influentes")
inf = influence_measures(fit)
cooks = inf["cooks"]
dffits = inf["dffits"]
dfbetas = inf["dfbetas"]
leverage = inf["leverage"]

cooks_df = pd.DataFrame({"index": df_mod.index, "cooks": cooks, "dffits": dffits, "leverage": leverage}).set_index("index").sort_values("cooks", ascending=False)
st.subheader("Top 10 Cook's distance")
st.dataframe(cooks_df.head(10).style.format("{:.6g}"))

st.subheader("Top 10 |DFFITS|")
st.dataframe(pd.DataFrame({"index": df_mod.index, "abs_dffits": np.abs(dffits)}).set_index("index").sort_values("abs_dffits", ascending=False).head(10).style.format("{:.6g}"))

dfbetas_df = pd.DataFrame(dfbetas, index=df_mod.index, columns=coef_names)
st.subheader("DFBETAS (max abs per coef)")
st.dataframe(dfbetas_df.abs().max().sort_values(ascending=False).to_frame("max_abs_dfbeta").style.format("{:.6g}"))

fig, ax = plt.subplots(figsize=(8,3))
ax.stem(np.arange(len(cooks)), cooks, markerfmt=",", basefmt=" ")
ax.set_xlabel("Observação")
ax.set_ylabel("Cook's distance")
st.pyplot(fig)

# 5) MÉTRICAS
st.header("5️⃣ Métricas do Modelo")
st.write(f"Observações: {fit['n']}  |  Parâmetros: {fit['p']}")
st.write(f"R² = {fit['R2']:.6f}  |  RMSE = {RMSE:.6f}")
st.write(f"AIC = {fit['aic']:.6g}  |  BIC = {fit['bic']:.6g}")
st.write(f"F-statistic (aprox) = {F_stat:.6g}  (p-values aproximados via |t|>~2)")

# 6) COMPARAÇÃO / VALIDAÇÃO CRUZADA (manual K-fold)
st.header("6️⃣ Comparação / Validação Cruzada (K-Fold manual)")
k = st.number_input("K folds", min_value=2, max_value=10, value=5)
def manual_cv_rmse(Xall, yall, kfolds=5, random_state=42):
    n = len(yall)
    idx = np.arange(n)
    rng = np.random.RandomState(random_state)
    rng.shuffle(idx)
    folds = np.array_split(idx, kfolds)
    rmses = []
    for i in range(kfolds):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(kfolds) if j!=i])
        Xtrain = Xall[train_idx]
        ytrain = yall[train_idx]
        Xtest = Xall[test_idx]
        ytest = yall[test_idx]
        fit_cv = ols_fit(Xtrain, ytrain, add_intercept=True)
        # predict manually: Xtest_design @ beta
        Xtest_design = np.column_stack([np.ones(len(Xtest)), Xtest])
        preds = Xtest_design @ fit_cv["beta"].reshape(-1,1)
        rmse = np.sqrt(np.mean((ytest.reshape(-1,1) - preds)**2))
        rmses.append(float(rmse))
    return rmses

with st.spinner("Executando CV..."):
    rmses_all = manual_cv_rmse(X_df.values, Y, kfolds=k)
    rmses_sel = manual_cv_rmse(X_sel.values, Y, kfolds=k)

st.write("RMSE CV (all predictors): mean={:.4f}, std={:.4f}".format(np.mean(rmses_all), np.std(rmses_all)))
st.write("RMSE CV (selected): mean={:.4f}, std={:.4f}".format(np.mean(rmses_sel), np.std(rmses_sel)))

st.write("AIC/BIC (modelo selecionado): AIC={:.4f}, BIC={:.4f}".format(fit["aic"], fit["bic"]))

# Classificação opcional (logistic manual)
st.subheader("Classificação (opcional) — Logistic via Newton-Raphson (numpy)")
do_class = st.checkbox("Transformar target em binário e treinar logistic", value=False)
if do_class:
    threshold = st.number_input("Threshold para classe positiva (Y >= )", value=int(np.nanmedian(Y)))
    y_bin = (Y >= threshold).astype(int)
    logfit = logistic_fit_newton(X_sel.values, y_bin, add_intercept=True, max_iter=100)
    proba = logfit["proba"]
    preds = (proba >= 0.5).astype(int)
    # metrics
    tp = int(((preds==1) & (y_bin==1)).sum())
    fp = int(((preds==1) & (y_bin==0)).sum())
    tn = int(((preds==0) & (y_bin==0)).sum())
    fn = int(((preds==0) & (y_bin==1)).sum())
    accuracy = (tp + tn) / max(len(y_bin), 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = roc_auc_from_probs(y_bin, proba)
    st.write(f"Acurácia: {accuracy:.4f}")
    st.write(f"Precisão: {precision:.4f}")
    st.write(f"Sensibilidade (Recall): {recall:.4f}")
    st.write(f"Especificidade: {specificity:.4f}")
    st.write(f"F1: {f1:.4f}")
    st.write(f"AUC: {auc:.4f}")
    # ROC plot
    desc = np.argsort(-proba)
    t = y_bin[desc]
    tp_cum = np.cumsum(t)
    fp_cum = np.cumsum(1 - t)
    tpr = tp_cum / max(tp_cum[-1], 1)
    fpr = fp_cum / max(fp_cum[-1], 1)
    plt.figure(figsize=(6,4))
    plt.plot(np.concatenate([[0], fpr]), np.concatenate([[0], tpr]), label=f"AUC={auc:.4f}")
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel("1 - Especificidade (FPR)")
    plt.ylabel("Sensibilidade (TPR)")
    plt.legend()
    st.pyplot(plt.gcf())

st.markdown("---")
st.write("Análise finalizada. Onde necessário mostramos estatísticas e regras práticas (ex.: |t|≈2 → p≈0.05) para interpretar sem funções de distribuição exatas.")
