# streamlit_enem_analysis.py
# Versão final: apenas streamlit, pandas, numpy, openpyxl, plotly
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import log

st.set_page_config(layout="wide", page_title="Análise ENEM (Plotly)")

st.title("📊 Análise Estatística e Modelagem — ENEM (Plotly, sem dependências nativas)")
st.markdown(
    "App compatível com Streamlit Cloud Free. Usa apenas numpy/pandas/plotly/openpyxl/streamlit."
)

# ------------------ utilitários numéricos (numpy) ------------------

def ols_fit(X, y, add_intercept=True):
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
    # protect against zero SE
    se_beta = np.where(se_beta == 0, 1e-12, se_beta)
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
    denom_safe = np.where((1 - h) == 0, 1e-12, (1 - h))
    cooks = (resid**2) / (p * MSE) * (h / (denom_safe**2))
    with np.errstate(divide='ignore', invalid='ignore'):
        dffits = (resid / np.sqrt(MSE * (1 - h))) * np.sqrt(h)
    invXtX_Xt = invXtX @ X.T
    delta_b = - (invXtX_Xt * resid) / denom_safe  # p x n
    se_beta = np.sqrt(np.diag(invXtX) * MSE).reshape(-1, 1)
    dfbetas = (delta_b / se_beta).T
    return {"leverage": h, "cooks": cooks, "dffits": dffits, "dfbetas": dfbetas}

def logistic_newton(X, y, add_intercept=True, max_iter=100, tol=1e-6):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    if add_intercept:
        X_design = np.column_stack([np.ones(len(X)), X])
    else:
        X_design = X
    n, p = X_design.shape
    beta = np.zeros((p,1))
    for _ in range(max_iter):
        z = X_design @ beta
        mu = 1.0 / (1.0 + np.exp(-z))
        W = (mu * (1 - mu)).flatten()
        W_safe = np.where(W == 0, 1e-12, W)
        grad = X_design.T @ (y - mu)
        XW = X_design * W_safe.reshape(-1,1)
        H = -(X_design.T @ XW)
        try:
            delta = np.linalg.pinv(H) @ grad
        except Exception:
            delta = np.linalg.pinv(H + 1e-6*np.eye(p)) @ grad
        beta_new = beta - delta
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    probs = (1.0 / (1.0 + np.exp(-(X_design @ beta)))).flatten()
    return {"beta": beta.flatten(), "proba": probs, "X_design": X_design}

def auc_from_probs(y_true, probs):
    y = np.asarray(y_true)
    p = np.asarray(probs)
    idx = np.argsort(-p)
    y_sorted = y[idx]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tp_rate = tp / max(tp[-1], 1)
    fp_rate = fp / max(fp[-1], 1)
    x = np.concatenate([[0.0], fp_rate])
    yv = np.concatenate([[0.0], tp_rate])
    auc = 0.0
    for i in range(1, len(x)):
        auc += (x[i] - x[i-1]) * (yv[i] + yv[i-1]) / 2.0
    return abs(auc)

# ------------------ UI / Entrada de dados ------------------

st.sidebar.header("Dados")
use_upload = st.sidebar.checkbox("Fazer upload do arquivo (.xlsx)", value=False)

if use_upload:
    uploaded = st.sidebar.file_uploader("Escolha o arquivo .xlsx", type=["xlsx"])
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

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("Nenhuma coluna numérica encontrada.")
    st.stop()

st.sidebar.header("Variáveis")
target = st.sidebar.selectbox("Escolha target (Y)", numeric_cols, index=0)
predictors = st.sidebar.multiselect("Escolha preditores (X) — vazio = todas as numéricas exceto Y",
                                    [c for c in numeric_cols if c != target])
if not predictors:
    predictors = [c for c in numeric_cols if c != target]

st.write("**Target:**", target)
st.write("**Preditores:**", predictors)

# filtrar NAs
df_mod = df[[target] + predictors].dropna()
Y = df_mod[target].values
X_df = df_mod[predictors].copy()

# ------------------ 1) Correlação ------------------
st.header("1️⃣ Análise de Correlação")
corr = df_mod.corr()
st.subheader("Matriz de Correlação (Pearson)")
st.dataframe(corr.style.background_gradient(cmap="RdBu").format(precision=3))

# p-values by permutation (cheap-ish default 200)
def pearson_perm_pvals(df_numeric, n_perm=200):
    cols = df_numeric.columns
    pmat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            x = df_numeric.iloc[:, i].values
            y = df_numeric.iloc[:, j].values
            robs = np.corrcoef(x, y)[0,1]
            count = 0
            for _ in range(n_perm):
                yperm = np.random.permutation(y)
                rperm = np.corrcoef(x, yperm)[0,1]
                if abs(rperm) >= abs(robs):
                    count += 1
            p = (count + 1) / (n_perm + 1)
            pmat.iloc[i,j] = p
            pmat.iloc[j,i] = p
    return pmat

st.subheader("P-valores (permutação, 200 permutações — reduzível)")
perm_proc = st.checkbox("Usar menos permutações (mais rápido)?", value=True)
n_perm = 100 if perm_proc else 300
with st.spinner("Calculando p-values por permutação..."):
    pvals = pearson_perm_pvals(df_mod, n_perm=n_perm)
st.dataframe(pvals.style.format(precision=4).applymap(lambda v: 'background-color: #ffcccc' if (isinstance(v, float) and v < 0.05) else ''))

# scatter interactive with plotly
st.subheader("Gráfico de Dispersão (interativo)")
scatter_feature = st.selectbox("Escolha um preditor para scatter", predictors)
fig_scatter = px.scatter(df_mod, x=scatter_feature, y=target, trendline="ols", title=f"{target} vs {scatter_feature}")
st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------ 2) Seleção de Variáveis ------------------
st.header("2️⃣ Seleção de Variáveis (forward / backward / stepwise)")
method = st.selectbox("Método", ["backward", "forward", "stepwise"])
alpha_in = st.number_input("p-valor para entrar (aprox via |t|~2)", value=0.05, format="%.4f")
alpha_out = st.number_input("p-valor para sair (aprox via |t|~2)", value=0.05, format="%.4f")

def stepwise_selection(Xdf, y, method="both", verbose=True):
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
                tvals = fit["t"][1:]
                p_approx_list = [0.05 if abs(t)>=2 else 0.32 for t in tvals]
                worst_idx = int(np.argmax(p_approx_list))
                if p_approx_list[worst_idx] > alpha_out:
                    rem = included[worst_idx]
                    included.remove(rem); changed=True
                    if verbose: st.write(f"Drop {rem} (approx p {p_approx_list[worst_idx]})")
        if not changed:
            break
    return included

with st.spinner("Executando seleção..."):
    selected = stepwise_selection(X_df, Y, method=method, verbose=True)
st.success(f"Variáveis selecionadas: {selected}")

# ------------------ Fit final OLS ------------------
st.header("Modelo Final (OLS — cálculos em numpy)")
X_sel = X_df[selected] if len(selected)>0 else X_df.copy()
fit = ols_fit(X_sel.values, Y, add_intercept=True)
coef_names = ["Intercept"] + list(X_sel.columns)
coef_df = pd.DataFrame({
    "coef": fit["beta"],
    "se": fit["se"],
    "t-stat": fit["t"]
}, index=coef_names)
st.subheader("Coeficientes (estimativa | se | t-stat)")
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
st.write(f"AIC = {fit['aic']:.6g}  |  BIC = {fit['bic']:.6g}")

# ------------------ 3) Diagnóstico ------------------
st.header("3️⃣ Diagnóstico das Suposições")

# Residuals vs Fitted (plotly)
fig = go.Figure()
fig.add_trace(go.Scatter(x=fit["y_hat"], y=fit["resid"], mode="markers", name="resíduos"))
fig.add_trace(go.Scatter(x=[min(fit["y_hat"]), max(fit["y_hat"])], y=[0,0], mode="lines", line=dict(color="red", dash="dash"), name="zero"))
fig.update_layout(title="Resíduos vs Valores Previstos", xaxis_title="y_hat", yaxis_title="resíduos")
st.plotly_chart(fig, use_container_width=True)

dw = durbin_watson(fit["resid"])
st.write(f"Durbin-Watson = {dw:.4f} (≈2 sem autocorrelação)")

bp = breusch_pagan_stat(fit["resid"], X_sel.values)
st.write(f"Breusch-Pagan LM = {bp['LM']:.6g}  — interprete comparando com chi2 critério (df = k)")

jb = jarque_bera(fit["resid"])
st.write(f"Jarque-Bera = {jb['JB']:.6g}; skew = {jb['skew']:.6g}; kurtosis = {jb['kurtosis']:.6g}")
st.info("Interpretação rápida: JB > 5.99 ⇒ rejeita normalidade ao nível 5% (aprox).")

vif_df = compute_vif(X_sel)
st.subheader("VIF (Multicolinearidade)")
st.dataframe(vif_df.style.format("{:.4f}"))

# ------------------ 4) Outliers / Influence ------------------
st.header("4️⃣ Outliers e Observações Influentes")
inf = influence_measures(fit)
cooks = inf["cooks"]
dffits = inf["dffits"]
dfbetas = inf["dfbetas"]
leverage = inf["leverage"]

cooks_df = pd.DataFrame({"index": df_mod.index, "cooks": cooks, "dffits": dffits, "leverage": leverage}).set_index("index").sort_values("cooks", ascending=False)
st.subheader("Top 10 — Cook's distance")
st.dataframe(cooks_df.head(10).style.format("{:.6g}"))

st.subheader("Top 10 — |DFFITS|")
st.dataframe(pd.DataFrame({"index": df_mod.index, "abs_dffits": np.abs(dffits)}).set_index("index").sort_values("abs_dffits", ascending=False).head(10).style.format("{:.6g}"))

dfbetas_df = pd.DataFrame(dfbetas, index=df_mod.index, columns=coef_names)
st.subheader("DFBETAS (máximo absoluto por coeficiente)")
st.dataframe(dfbetas_df.abs().max().sort_values(ascending=False).to_frame("max_abs_dfbeta").style.format("{:.6g}"))

# Cook's plot (plotly)
fig = px.bar(x=np.arange(len(cooks)), y=cooks, labels={"x":"índice", "y":"Cook's D"}, title="Cook's Distance")
st.plotly_chart(fig, use_container_width=True)

# ------------------ 5) Métricas do modelo ------------------
st.header("5️⃣ Métricas do Modelo")
st.write(f"Observações: {fit['n']}  |  Parâmetros (p): {fit['p']}")
st.write(f"R² = {fit['R2']:.6f}  |  RMSE = {RMSE:.6f}")
st.write(f"AIC = {fit['aic']:.6g}  |  BIC = {fit['bic']:.6g}")
st.write(f"F-statistic = {F_stat:.6g}  (p-values aproximados via |t|>~2)")

# ------------------ 6) Comparação / CV / Classificação ------------------
st.header("6️⃣ Comparação e Validação Cruzada")

k = st.number_input("K folds (CV)", min_value=2, max_value=10, value=5)
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
        Xtest_design = np.column_stack([np.ones(len(Xtest)), Xtest])
        preds = (Xtest_design @ fit_cv["beta"].reshape(-1,1)).flatten()
        rmse = np.sqrt(np.mean((ytest - preds)**2))
        rmses.append(float(rmse))
    return rmses

with st.spinner("Executando CV..."):
    rmses_all = manual_cv_rmse(X_df.values, Y, kfolds=k)
    rmses_sel = manual_cv_rmse(X_sel.values, Y, kfolds=k)

st.write("RMSE CV — All preds: mean={:.4f}, std={:.4f}".format(np.mean(rmses_all), np.std(rmses_all)))
st.write("RMSE CV — Selected: mean={:.4f}, std={:.4f}".format(np.mean(rmses_sel), np.std(rmses_sel)))

st.subheader("Classificação (opcional)")
do_class = st.checkbox("Transformar target em binário e treinar logistic (Newton-Raphson)", value=False)
if do_class:
    threshold = st.number_input("Threshold para classe positiva (Y >= )", value=int(np.nanmedian(Y)))
    y_bin = (Y >= threshold).astype(int)
    logfit = logistic_newton(X_sel.values, y_bin, add_intercept=True, max_iter=200)
    probs = logfit["proba"]
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds==1) & (y_bin==1)).sum())
    fp = int(((preds==1) & (y_bin==0)).sum())
    tn = int(((preds==0) & (y_bin==0)).sum())
    fn = int(((preds==0) & (y_bin==1)).sum())
    accuracy = (tp + tn) / max(len(y_bin), 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = auc_from_probs(y_bin, probs)
    st.write(f"Acurácia: {accuracy:.4f}")
    st.write(f"Precisão: {precision:.4f}")
    st.write(f"Sensibilidade (Recall): {recall:.4f}")
    st.write(f"Especificidade: {specificity:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"AUC: {auc:.4f}")
    # ROC plot
    idx = np.argsort(-probs)
    y_sorted = y_bin[idx]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([[0], fpr]), y=np.concatenate([[0], tpr]), mode='lines', name=f"AUC={auc:.4f}"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='red'), showlegend=False))
    fig.update_layout(title="ROC Curve", xaxis_title="1 - Especificidade (FPR)", yaxis_title="Sensibilidade (TPR)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("Análise concluída. Observações: onde p-values exatos exigiriam funções de distribuição, o app mostra estatísticas e fornece regras práticas de interpretação (ex.: |t| ≳ 2 ⇒ p≈0.05 para df grande; JB > 5.99 ⇒ rejeita normalidade a 5%).")
