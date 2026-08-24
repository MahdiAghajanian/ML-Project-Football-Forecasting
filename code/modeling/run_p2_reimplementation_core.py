#!/usr/bin/env python
"""Full evaluation for the P2 project reimplementation.

The official PyPI NGBoost artifacts are used only for an optional, explicitly
labelled comparison at the end. They never define P2 features, model selection,
calibration, or predictions.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from scipy.stats import chi2, norm, pearsonr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if not (HERE / "p2_reimplementation").exists():
    import subprocess
    subprocess.check_call([sys.executable, str(HERE / "prepare_p2_reimplementation.py")])

from p2_reimplementation import NGBClassifier, NGBRegressor
from p2_reimplementation.distns import MultivariateNormal, Normal, k_categorical
from p2_reimplementation.scores import LogScore

SEED = 42
SPLITS = {"train": 251, "validation": 77, "test": 46}
CLASS = {"A": 0, "D": 1, "H": 2}
NAMES = ["Away", "Draw", "Home"]


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "standard", "full"], default="full")
    p.add_argument("--project-root", default="")
    p.add_argument("--skip-multivariate", action="store_true")
    return p.parse_args()


def normalize(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def encode(s):
    y = s.map(CLASS)
    if y.isna().any():
        raise ValueError("Outcome labels must be A/D/H")
    return y.to_numpy(int)


def onehot(y):
    return np.eye(3)[np.asarray(y, int)]


def row_rps(y, p):
    p, o = normalize(p), onehot(y)
    return np.sum((np.cumsum(p, 1)[:, :-1] - np.cumsum(o, 1)[:, :-1]) ** 2, axis=1) / 2


def reliability(y, p, bins=10):
    p = normalize(p); y = np.asarray(y); conf = p.max(1); pred = p.argmax(1); ok = pred == y
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & (conf < hi if b < bins - 1 else conf <= hi)
        rows.append({"bin": b, "lower": lo, "upper": hi, "count": int(m.sum()),
                     "mean_confidence": float(conf[m].mean()) if m.any() else np.nan,
                     "accuracy": float(ok[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def cmetrics(y, p):
    p = normalize(p); pred = p.argmax(1)
    pr, re, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    rel = reliability(y, p); n = max(1, rel["count"].sum())
    ece = float(np.nansum(rel["count"] / n * np.abs(rel["accuracy"] - rel["mean_confidence"])))
    d = {"RPS": float(row_rps(y, p).mean()), "LogLoss": float(log_loss(y, p, labels=[0, 1, 2])),
         "Brier": float(np.mean(np.sum((p - onehot(y)) ** 2, axis=1))), "ECE": ece,
         "Accuracy": float(accuracy_score(y, pred)), "MacroPrecision": float(pr.mean()),
         "MacroRecall": float(re.mean()), "MacroF1": float(f1.mean())}
    for i, name in enumerate(NAMES):
        d[f"Precision_{name}"] = float(pr[i]); d[f"Recall_{name}"] = float(re[i]); d[f"F1_{name}"] = float(f1[i])
    return d


def rmetrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    corr = np.nan if len(y) < 2 or np.std(y) == 0 or np.std(pred) == 0 else float(pearsonr(y, pred).statistic)
    return {"MAE": float(mean_absolute_error(y, pred)), "RMSE": float(mean_squared_error(y, pred) ** .5), "Correlation": corr}


def coverage(y, mu, sig, levels=(.5, .8, .95)):
    y, mu, sig = np.asarray(y, float), np.asarray(mu, float), np.maximum(np.asarray(sig, float), 1e-8)
    out = {}
    for level in levels:
        z = norm.ppf((1 + level) / 2); inside = np.abs(y - mu) <= z * sig
        out[f"Coverage{int(level*100)}"] = float(inside.mean())
        out[f"MeanWidth{int(level*100)}"] = float(np.mean(2 * z * sig))
    return out


class Platt:
    def __init__(self):
        self.model = LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)
    def fit(self, p, y):
        self.model.fit(np.log(normalize(p)), y); return self
    def predict_proba(self, p):
        return normalize(self.model.predict_proba(np.log(normalize(p))))


def pipe(model):
    return Pipeline([("prep", SimpleImputer(strategy="median", add_indicator=True)), ("model", model)])


def pdist(model, X):
    return model.named_steps["model"].pred_dist(model.named_steps["prep"].transform(X))


def candidates(live=False, reg=False, mv=False):
    ns, rates, leaves = [250, 400, 600], [.02, .03, .05], [12, 8, 5]
    depths = [3, 4, 5] if live else [2, 3, 4]
    out = []
    for i in range(3):
        base = DecisionTreeRegressor(max_depth=depths[i], min_samples_leaf=leaves[i], random_state=SEED)
        kw = dict(Score=LogScore, Base=base, n_estimators=ns[i], learning_rate=rates[i],
                  minibatch_frac=.8, col_sample=.8, natural_gradient=True, random_state=SEED, verbose=False)
        if mv: model = NGBRegressor(Dist=MultivariateNormal(2), **kw)
        elif reg: model = NGBRegressor(Dist=Normal, **kw)
        else: model = NGBClassifier(Dist=k_categorical(3), **kw)
        out.append(pipe(model))
    return out


class RSSMonitor:
    def __init__(self, interval=.01): self.interval = interval
    def __enter__(self):
        self.proc = psutil.Process(os.getpid()); self.base = self.proc.memory_info().rss; self.peak = self.base
        self.stop = threading.Event()
        def loop():
            while not self.stop.wait(self.interval):
                try: self.peak = max(self.peak, self.proc.memory_info().rss)
                except psutil.Error: pass
        self.thread = threading.Thread(target=loop, daemon=True); self.thread.start(); return self
    def __exit__(self, *_):
        self.stop.set(); self.thread.join(timeout=1)
        try: self.peak = max(self.peak, self.proc.memory_info().rss)
        except psutil.Error: pass


def fit_measured(est, X, y):
    t = time.perf_counter()
    with RSSMonitor() as mon: fitted = clone(est).fit(X, y)
    return fitted, {"TrainSeconds": time.perf_counter() - t, "BaselineRSSMB": mon.base / 2**20,
                    "PeakRSSMB": mon.peak / 2**20, "PeakDeltaMB": (mon.peak - mon.base) / 2**20}


def latency(model, X, reps=30):
    Xs = X.iloc[:min(64, len(X))] if hasattr(X, "iloc") else X[:min(64, len(X))]; vals = []
    for _ in range(reps):
        t = time.perf_counter()
        if hasattr(model, "predict_proba"): model.predict_proba(Xs)
        else: model.predict(Xs)
        vals.append((time.perf_counter() - t) * 1000)
    return {"PredictBatchRows": len(Xs), "PredictP50ms": float(np.percentile(vals, 50)),
            "PredictP95ms": float(np.percentile(vals, 95)), "PredictP99ms": float(np.percentile(vals, 99))}


def choose(cands, Xt, yt, Xv, yv, task, limit, label):
    rows, fits = [], []
    for i, est in enumerate(cands[:limit], 1):
        m, res = fit_measured(est, Xt, yt)
        if task == "classification": crit, score = "validation_RPS", float(row_rps(yv, m.predict_proba(Xv)).mean())
        else: crit, score = "validation_NLL", float(-np.mean(pdist(m, Xv).logpdf(yv)))
        rows.append({"Model": label, "Candidate": i, "SelectionCriterion": crit, crit: score,
                     **res, "Parameters": repr(est.get_params(deep=True))}); fits.append(m)
    best = int(np.argmin([r[r["SelectionCriterion"]] for r in rows]))
    for i, r in enumerate(rows): r["Selected"] = i == best
    return fits[best], pd.DataFrame(rows), rows[best]


def root_of(override):
    rel = Path("code/data_pipeline/results/Premier_League_2015_16_DATA_ONLY_FULL_REAL/data/gold/gold_prematch.parquet")
    cand = ([Path(override)] if override else []) + [Path.cwd(), Path.cwd().parent,
        Path("/content/ML-Project-Football-Forecasting"), Path("/content/drive/MyDrive/ml_project"),
        Path("/content/drive/MyDrive/ML-Project-Football-Forecasting")]
    for p in cand:
        if (p / rel).exists(): return p.resolve()
    raise FileNotFoundError("Project root not found; pass --project-root")


def load(root):
    result = root / "code/data_pipeline/results/Premier_League_2015_16_DATA_ONLY_FULL_REAL"
    gold, audit = result / "data/gold", result / "audit"
    p1path = root / "code/data_pipeline/results/EXP01_P1_REPRESENTATION_TASK_C/data/prematch_md1_plus_p1_homeaway.parquet"
    pre = pd.read_parquet(gold / "gold_prematch.parquet").sort_values("kickoff").reset_index(drop=True)
    live = pd.read_parquet(gold / "gold_snapshots.parquet").sort_values(["match_id", "snapshot_rank"]).reset_index(drop=True)
    p1all = pd.read_parquet(p1path); p1f = [c for c in p1all.columns if c.startswith("p1_")]
    fd = pd.read_csv(audit / "feature_distribution_summary.csv")
    md1pre = [c for c in fd.loc[fd.table.eq("gold_prematch"), "feature"].drop_duplicates() if c in pre.columns and not c.startswith("label_")]
    md1live = [c for c in fd.loc[fd.table.eq("gold_snapshots"), "feature"].drop_duplicates() if c in live.columns and not c.startswith("label_")]
    if (len(md1pre), len(md1live), len(p1f)) != (128, 189, 18):
        raise RuntimeError(f"Unexpected feature contract: {len(md1pre)}/{len(md1live)}/{len(p1f)}")
    payload = p1all[["match_id", *p1f]]
    pre = pre.merge(payload, on="match_id", validate="one_to_one").sort_values("kickoff").reset_index(drop=True)
    live = live.merge(payload, on="match_id", validate="many_to_one").sort_values(["match_id", "snapshot_rank"]).reset_index(drop=True)
    if pre.groupby("split").size().to_dict() != SPLITS or not live.groupby("match_id").size().eq(21).all():
        raise RuntimeError("Frozen split/snapshot contract failed")
    pf, lf = md1pre + p1f, md1live + p1f
    ps = {s: pre[pre.split.eq(s)].copy().reset_index(drop=True) for s in SPLITS}
    ls = {s: live[live.split.eq(s)].copy().reset_index(drop=True) for s in SPLITS}
    Xp = {s: f[pf] for s, f in ps.items()}; Xl = {s: f[lf] for s, f in ls.items()}
    yc = {s: encode(f.label_outcome) for s, f in ps.items()}; yr = {s: f.label_margin.to_numpy(float) for s, f in ps.items()}
    ylc = {s: encode(f.label_outcome) for s, f in ls.items()}; ylr = {s: f.label_margin.to_numpy(float) for s, f in ls.items()}
    contract = {"source": "data-pipeline feature_distribution_summary.csv + P1 parquet schema",
                "md1_pre_features": md1pre, "p1_features": p1f, "pre_features": pf,
                "md1_live_features": md1live, "live_features": lf}
    return contract, ps, ls, Xp, yc, yr, Xl, ylc, ylr


def phase(m):
    m = float(m)
    return "0-15" if m <= 15 else "15-30" if m <= 30 else "30-45/HT" if m <= 45.5 else "HT-60" if m <= 60 else "60-75" if m <= 75 else "75-FT"


def plot_reliability(df, path, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.5)); ax.plot([0,1],[0,1], "--", linewidth=1)
    for label, g in df.groupby("Calibration"):
        g = g.dropna(subset=["mean_confidence", "accuracy"]); ax.plot(g.mean_confidence, g.accuracy, marker="o", label=label)
    ax.set(xlabel="Mean confidence", ylabel="Empirical accuracy", title=title, xlim=(0,1), ylim=(0,1)); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def mv_metrics(Y, means, covs, seed=SEED, samples_n=800):
    Y, means, covs = np.asarray(Y,float), np.asarray(means,float), np.asarray(covs,float)
    inv = np.linalg.inv(covs); delta = Y-means; md2 = np.einsum("ni,nij,nj->n", delta, inv, delta)
    eig = np.linalg.eigvalsh(covs); sd = np.sqrt(np.diagonal(covs, axis1=1, axis2=2)); corr = covs[:,0,1]/np.sqrt(covs[:,0,0]*covs[:,1,1])
    nll = float(np.mean(.5*(2*np.log(2*np.pi)+np.linalg.slogdet(covs)[1]+md2)))
    rng=np.random.default_rng(seed); sam=np.stack([rng.multivariate_normal(means[i],covs[i],size=samples_n) for i in range(len(Y))]); half=samples_n//2
    energy=float(np.mean(np.linalg.norm(sam-Y[:,None,:],axis=2).mean(1)-.5*np.linalg.norm(sam[:,:half]-sam[:,half:2*half],axis=2).mean(1)))
    rounded=np.maximum(np.rint(sam),0).astype(int); outc=np.where(rounded[:,:,0]>rounded[:,:,1],2,np.where(rounded[:,:,0]<rounded[:,:,1],0,1)); probs=np.stack([(outc==k).mean(1) for k in [0,1,2]],axis=1); yout=np.where(Y[:,0]>Y[:,1],2,np.where(Y[:,0]<Y[:,1],0,1))
    d={"HomeGoalsRMSE":float(mean_squared_error(Y[:,0],means[:,0])**.5),"AwayGoalsRMSE":float(mean_squared_error(Y[:,1],means[:,1])**.5),"JointNLL":nll,"EnergyScore":energy,
       "MeanPredictedCorrelation":float(np.mean(corr)),"MinCovarianceEigenvalue":float(eig.min()),"MedianCovarianceEigenvalue":float(np.median(eig)),
       "HomeGoalCoverage80":float((np.abs(Y[:,0]-means[:,0])<=norm.ppf(.9)*sd[:,0]).mean()),"AwayGoalCoverage80":float((np.abs(Y[:,1]-means[:,1])<=norm.ppf(.9)*sd[:,1]).mean()),
       "JointEllipseCoverage80":float((md2<=chi2.ppf(.8,df=2)).mean())}
    d.update({f"Outcome_{k}":v for k,v in cmetrics(yout,probs).items()}); return d, probs, eig, corr


def main():
    a=cli(); np.random.seed(SEED); root=root_of(a.project_root); limit={"quick":1,"standard":2,"full":3}[a.mode]
    out=root/f"code/modeling/outputs/p2_reimplementation_{a.mode}"; tab,mdl,figdir=out/"tables",out/"models",out/"figures"
    for d in (tab,mdl,figdir): d.mkdir(parents=True,exist_ok=True)
    contract,ps,ls,Xp,yc,yr,Xl,ylc,ylr=load(root); (out/"feature_contract.json").write_text(json.dumps(contract,indent=2),encoding="utf-8")
    summary={}; compute=[]

    # Task C
    cm,ct,cres=choose(candidates(),Xp["train"],yc["train"],Xp["validation"],yc["validation"],"classification",limit,"P2 Task C")
    cal=Platt().fit(cm.predict_proba(Xp["validation"]),yc["validation"]); raw=cm.predict_proba(Xp["test"]); prob=cal.predict_proba(raw)
    ctab=pd.DataFrame([{"Calibration":"Raw",**cmetrics(yc["test"],raw)},{"Calibration":"Platt",**cmetrics(yc["test"],prob)}]); ctab.to_csv(tab/"p2_task_c_metrics.csv",index=False); ct.to_csv(tab/"p2_task_c_tuning.csv",index=False)
    rows=ps["test"][["match_id","label_outcome"]].copy(); rows["y_int"]=yc["test"]
    for tag,p in [("raw",raw),("calibrated",prob)]:
        for i,n in enumerate(NAMES): rows[f"p_{n.lower()}_{tag}"]=p[:,i]
        rows[f"rps_{tag}"]=row_rps(yc["test"],p)
    rows.to_csv(tab/"p2_task_c_predictions.csv",index=False); rows.nlargest(10,"rps_calibrated").to_csv(tab/"p2_task_c_worst10.csv",index=False)
    rel=pd.concat([reliability(yc["test"],raw).assign(Calibration="Raw"),reliability(yc["test"],prob).assign(Calibration="Platt")]); rel.to_csv(tab/"p2_task_c_reliability.csv",index=False); plot_reliability(rel,figdir/"p2_task_c_reliability.png","P2 Task C reliability")
    for tag,p in [("raw",raw),("platt",prob)]: pd.DataFrame(confusion_matrix(yc["test"],p.argmax(1),labels=[0,1,2]),index=NAMES,columns=NAMES).to_csv(tab/f"p2_task_c_confusion_{tag}.csv")
    summary["task_c"]={r.Calibration:r.drop(labels="Calibration").to_dict() for _,r in ctab.iterrows()}; compute.append({**cres,**latency(cm,Xp["test"])}); joblib.dump(cm,mdl/"p2_task_c_classifier.joblib"); joblib.dump(cal,mdl/"p2_task_c_platt.joblib")

    # Task R
    rm,rt,rres=choose(candidates(reg=True),Xp["train"],yr["train"],Xp["validation"],yr["validation"],"regression",limit,"P2 Task R")
    vd=pdist(rm,Xp["validation"]); td=pdist(rm,Xp["test"]); vmu=rm.predict(Xp["validation"]); mu=rm.predict(Xp["test"]); vs=np.maximum(np.asarray(vd.params["scale"],float),1e-8); sig0=np.maximum(np.asarray(td.params["scale"],float),1e-8); sf=float(np.sqrt(np.mean(((yr["validation"]-vmu)/vs)**2))); sig=sig0*sf
    rs={**rmetrics(yr["test"],mu),"ScaleFactorFromValidation":sf,"TestNLLRaw":float(-np.mean(td.logpdf(yr["test"]))),"TestNLLCalibrated":float(-np.mean(norm.logpdf(yr["test"],loc=mu,scale=sig))),**{f"Raw_{k}":v for k,v in coverage(yr["test"],mu,sig0).items()},**{f"Calibrated_{k}":v for k,v in coverage(yr["test"],mu,sig).items()}}
    pd.DataFrame([rs]).to_csv(tab/"p2_task_r_metrics.csv",index=False); rt.to_csv(tab/"p2_task_r_tuning.csv",index=False)
    rr=ps["test"][["match_id","label_margin"]].copy(); rr["pred_mean"]=mu; rr["sigma_raw"]=sig0; rr["sigma_calibrated"]=sig; rr["abs_error"]=np.abs(rr.label_margin-mu)
    for level in (.5,.8,.95): z=norm.ppf((1+level)/2); rr[f"lower{int(level*100)}"]=mu-z*sig; rr[f"upper{int(level*100)}"]=mu+z*sig
    rr.to_csv(tab/"p2_task_r_predictions.csv",index=False); rr.nlargest(10,"abs_error").to_csv(tab/"p2_task_r_worst10.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,4)); order=np.argsort(mu); x=np.arange(len(mu)); ax.errorbar(x,mu[order],yerr=norm.ppf(.9)*sig[order],fmt="o",markersize=3,alpha=.7); ax.scatter(x,yr["test"][order],s=10,label="Actual"); ax.set(title="P2 Task R 80% predictive intervals",xlabel="Test matches sorted by predicted margin",ylabel="Goal margin"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"p2_task_r_intervals.png",dpi=180); plt.close(fig)
    summary["task_r"]=rs; compute.append({**rres,**latency(rm,Xp["test"])}); joblib.dump(rm,mdl/"p2_task_r_regressor.joblib")

    # Frozen P2 pre-match baselines for every live test snapshot.
    frozenp=pd.DataFrame(prob,columns=["p0","p1","p2"]); frozenp["match_id"]=ps["test"].match_id.to_numpy(); frozenm=pd.DataFrame({"match_id":ps["test"].match_id,"mu":mu,"sigma":sig})
    fpc=ls["test"][["match_id"]].merge(frozenp,on="match_id",validate="many_to_one")[["p0","p1","p2"]].to_numpy(); fmr=ls["test"][["match_id"]].merge(frozenm,on="match_id",validate="many_to_one")

    # Task L classification
    lm,lct,lcres=choose(candidates(live=True),Xl["train"],ylc["train"],Xl["validation"],ylc["validation"],"classification",limit,"P2 Task L outcome")
    lcal=Platt().fit(lm.predict_proba(Xl["validation"]),ylc["validation"]); lraw=lm.predict_proba(Xl["test"]); lprob=lcal.predict_proba(lraw)
    lctab=pd.DataFrame([{"Evaluation":"Live Raw",**cmetrics(ylc["test"],lraw)},{"Evaluation":"Live Platt",**cmetrics(ylc["test"],lprob)},{"Evaluation":"Frozen P2 Pre-match",**cmetrics(ylc["test"],fpc)}]); lctab.to_csv(tab/"p2_task_l_outcome_metrics.csv",index=False); lct.to_csv(tab/"p2_task_l_outcome_tuning.csv",index=False)
    lrows=ls["test"][[c for c in ["match_id","snapshot_id","snapshot_minute","snapshot_rank","label_outcome"] if c in ls["test"].columns]].copy(); lrows["y_int"]=ylc["test"]
    for tag,p in [("raw",lraw),("calibrated",lprob),("frozen",fpc)]:
        for i,n in enumerate(NAMES): lrows[f"p_{n.lower()}_{tag}"]=p[:,i]
        lrows[f"rps_{tag}"]=row_rps(ylc["test"],p)
    lrows["phase"]=lrows.snapshot_minute.map(phase); lrows.to_csv(tab/"p2_task_l_outcome_predictions.csv",index=False); lrows.nlargest(10,"rps_calibrated").to_csv(tab/"p2_task_l_outcome_worst10.csv",index=False)
    minute=[]; phase_rows=[]; phase_rel=[]
    for rank,g in lrows.groupby("snapshot_rank",sort=True):
        pos=g.index.to_numpy(); minute_value=float(g["snapshot_minute"].median())
        for label,p in [("Live Raw",lraw),("Live Platt",lprob),("Frozen P2 Pre-match",fpc)]: minute.append({"snapshot_rank":int(rank),"snapshot_minute":minute_value,"Evaluation":label,"Rows":len(pos),**cmetrics(ylc["test"][pos],p[pos])})
    for ph,g in lrows.groupby("phase",sort=False):
        pos=g.index.to_numpy()
        for label,p in [("Live Raw",lraw),("Live Platt",lprob),("Frozen P2 Pre-match",fpc)]:
            phase_rows.append({"Phase":ph,"Evaluation":label,"Rows":len(pos),**cmetrics(ylc["test"][pos],p[pos])})
            phase_rel.append(reliability(ylc["test"][pos],p[pos]).assign(Phase=ph,Evaluation=label))
    mdf=pd.DataFrame(minute); mdf.to_csv(tab/"p2_task_l_outcome_by_minute.csv",index=False); pd.DataFrame(phase_rows).to_csv(tab/"p2_task_l_outcome_by_phase.csv",index=False); pd.concat(phase_rel).to_csv(tab/"p2_task_l_phase_reliability.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,4));
    for label,g in mdf.groupby("Evaluation"): g=g.sort_values("snapshot_rank"); ax.plot(g.snapshot_minute,g.RPS,marker="o",label=label)
    ax.set(title="P2 Task L outcome RPS vs minute",xlabel="Minute",ylabel="RPS"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"p2_task_l_outcome_rps_vs_minute.png",dpi=180); plt.close(fig)
    summary["task_l_classification"]={r.Evaluation:r.drop(labels="Evaluation").to_dict() for _,r in lctab.iterrows()}; compute.append({**lcres,**latency(lm,Xl["test"])}); joblib.dump(lm,mdl/"p2_task_l_classifier.joblib"); joblib.dump(lcal,mdl/"p2_task_l_platt.joblib")

    # Task L regression
    lrm,lrt,lrres=choose(candidates(live=True,reg=True),Xl["train"],ylr["train"],Xl["validation"],ylr["validation"],"regression",limit,"P2 Task L margin")
    lvd=pdist(lrm,Xl["validation"]); ltd=pdist(lrm,Xl["test"]); lvmu=lrm.predict(Xl["validation"]); lmu=lrm.predict(Xl["test"]); lvs=np.maximum(np.asarray(lvd.params["scale"],float),1e-8); lsig0=np.maximum(np.asarray(ltd.params["scale"],float),1e-8); lsf=float(np.sqrt(np.mean(((ylr["validation"]-lvmu)/lvs)**2))); lsig=lsig0*lsf
    live_reg={"Evaluation":"Live",**rmetrics(ylr["test"],lmu),"NLL":float(-np.mean(norm.logpdf(ylr["test"],lmu,lsig))),**coverage(ylr["test"],lmu,lsig)}; frozen_reg={"Evaluation":"Frozen P2 Pre-match",**rmetrics(ylr["test"],fmr.mu),"NLL":float(-np.mean(norm.logpdf(ylr["test"],fmr.mu,fmr.sigma))),**coverage(ylr["test"],fmr.mu,fmr.sigma)}
    pd.DataFrame([live_reg,frozen_reg]).to_csv(tab/"p2_task_l_margin_metrics.csv",index=False); lrt.to_csv(tab/"p2_task_l_margin_tuning.csv",index=False)
    lrrows=ls["test"][[c for c in ["match_id","snapshot_id","snapshot_minute","snapshot_rank","label_margin"] if c in ls["test"].columns]].copy(); lrrows["phase"]=lrrows.snapshot_minute.map(phase); lrrows["live_mean"]=lmu; lrrows["live_sigma"]=lsig; lrrows["frozen_mean"]=fmr.mu.to_numpy(); lrrows["frozen_sigma"]=fmr.sigma.to_numpy(); lrrows["live_abs_error"]=np.abs(ylr["test"]-lmu); lrrows.to_csv(tab/"p2_task_l_margin_predictions.csv",index=False); lrrows.nlargest(10,"live_abs_error").to_csv(tab/"p2_task_l_margin_worst10.csv",index=False)
    mr=[]; pr=[]
    for rank,g in lrrows.groupby("snapshot_rank",sort=True):
        pos=g.index.to_numpy(); minute_value=float(g["snapshot_minute"].median())
        for label,pm,sg in [("Live",lmu,lsig),("Frozen P2 Pre-match",fmr.mu.to_numpy(),fmr.sigma.to_numpy())]: mr.append({"snapshot_rank":int(rank),"snapshot_minute":minute_value,"Evaluation":label,"Rows":len(pos),**rmetrics(ylr["test"][pos],pm[pos]),"NLL":float(-np.mean(norm.logpdf(ylr["test"][pos],pm[pos],sg[pos]))),**coverage(ylr["test"][pos],pm[pos],sg[pos])})
    for ph,g in lrrows.groupby("phase",sort=False):
        pos=g.index.to_numpy()
        for label,pm,sg in [("Live",lmu,lsig),("Frozen P2 Pre-match",fmr.mu.to_numpy(),fmr.sigma.to_numpy())]: pr.append({"Phase":ph,"Evaluation":label,"Rows":len(pos),**rmetrics(ylr["test"][pos],pm[pos]),"NLL":float(-np.mean(norm.logpdf(ylr["test"][pos],pm[pos],sg[pos]))),**coverage(ylr["test"][pos],pm[pos],sg[pos])})
    mrdf=pd.DataFrame(mr); mrdf.to_csv(tab/"p2_task_l_margin_by_minute.csv",index=False); pd.DataFrame(pr).to_csv(tab/"p2_task_l_margin_by_phase.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,4));
    for label,g in mrdf.groupby("Evaluation"): g=g.sort_values("snapshot_rank"); ax.plot(g.snapshot_minute,g.RMSE,marker="o",label=label)
    ax.set(title="P2 Task L margin RMSE vs minute",xlabel="Minute",ylabel="RMSE"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"p2_task_l_margin_rmse_vs_minute.png",dpi=180); plt.close(fig)
    summary["task_l_regression"]={"Live":live_reg,"Frozen":frozen_reg,"ValidationScaleFactor":lsf}; compute.append({**lrres,**latency(lrm,Xl["test"])}); joblib.dump(lrm,mdl/"p2_task_l_regressor.joblib")

    # P2 multivariate experiment
    if not a.skip_multivariate:
        Y={s:ps[s][["home_score","away_score"]].to_numpy(float) for s in SPLITS}
        mv,mvt,mvres=choose(candidates(mv=True),Xp["train"],Y["train"],Xp["validation"],Y["validation"],"regression",limit,"P2 multivariate")
        mvt.to_csv(tab/"p2_multivariate_tuning.csv",index=False); vd=pdist(mv,Xp["validation"]); td=pdist(mv,Xp["test"]); vmean=np.asarray(vd.loc); vcov=np.asarray(vd.cov); tmean=np.asarray(td.loc); tcov=np.asarray(td.cov)
        delta=Y["validation"]-vmean; vmd2=np.einsum("ni,nij,nj->n",delta,np.linalg.inv(vcov),delta); cov_scale=float(np.mean(vmd2)/2.0); ccov=tcov*cov_scale
        rawmv,rawprob,eig,corr=mv_metrics(Y["test"],tmean,tcov); calmv,calprob,ceig,ccorr=mv_metrics(Y["test"],tmean,ccov)
        pd.DataFrame([{"Calibration":"Raw",**rawmv},{"Calibration":"Validation dispersion",**calmv,"CovarianceScale":cov_scale}]).to_csv(tab/"p2_multivariate_metrics.csv",index=False)
        mvr=ps["test"][["match_id","home_score","away_score"]].copy(); mvr[["mean_home","mean_away"]]=tmean; mvr["corr_raw"]=corr
        for i,n in enumerate(NAMES): mvr[f"p_{n.lower()}_raw"]=rawprob[:,i]; mvr[f"p_{n.lower()}_calibrated"]=calprob[:,i]
        mvr.to_csv(tab/"p2_multivariate_predictions.csv",index=False)
        fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(np.arange(len(corr)),corr,s=14); ax.axhline(0,linewidth=1); ax.set(title="P2 predicted home-away goal correlation",xlabel="Test match index",ylabel="Predicted correlation"); fig.tight_layout(); fig.savefig(figdir/"p2_multivariate_predicted_correlation.png",dpi=180); plt.close(fig)
        summary["multivariate"]={"Raw":rawmv,"Calibrated":calmv,"CovarianceScale":cov_scale}; compute.append({**mvres,**latency(mv,Xp["test"])}); joblib.dump(mv,mdl/"p2_multivariate_goals.joblib")

    # Explicit official-library comparison only. The finalizer rebuilds the
    # complete C/R/L comparison and records any baseline-read errors.
    old=root/"code/modeling/outputs/ngboost_p1_corrected_full/tables"; comps=[]
    if old.exists():
        try:
            oc=pd.read_csv(old/"task_c_all_metrics.csv"); v=float(oc[(oc.Model=="NGBoost")&(oc.Calibration=="Platt")].iloc[0].RPS); comps += [{"Task":"C","Implementation":"Official NGBoost library baseline","Metric":"RPS","Value":v},{"Task":"C","Implementation":"P2 project reimplementation","Metric":"RPS","Value":summary["task_c"]["Platt"]["RPS"]}]
            orr=pd.read_csv(old/"task_r_all_metrics.csv"); v=float(orr[orr.Model=="NGBoost"].iloc[0].RMSE); comps += [{"Task":"R","Implementation":"Official NGBoost library baseline","Metric":"RMSE","Value":v},{"Task":"R","Implementation":"P2 project reimplementation","Metric":"RMSE","Value":summary["task_r"]["RMSE"]}]
            if "multivariate" in summary and (old/"p2_multivariate_ngboost_metrics.csv").exists():
                ov=float(pd.read_csv(old/"p2_multivariate_ngboost_metrics.csv").iloc[0].JointNLL); comps += [{"Task":"P2 multivariate","Implementation":"Official NGBoost library baseline","Metric":"JointNLL","Value":ov},{"Task":"P2 multivariate","Implementation":"P2 project reimplementation","Metric":"JointNLL","Value":summary["multivariate"]["Raw"]["JointNLL"]}]
        except Exception as exc: (out/"official_baseline_comparison_warning.txt").write_text(str(exc),encoding="utf-8")
    pd.DataFrame(comps).to_csv(tab/"p2_vs_official_library.csv",index=False); pd.DataFrame(compute).to_csv(tab/"p2_compute_and_peak_memory.csv",index=False)

    cfg={"seed":SEED,"mode":a.mode,"implementation":"P2 project reimplementation","package":"code/modeling/ngboost facade -> code/modeling/p2_source","external_ngboost_runtime_dependency":False,"feature_contract_source":contract["source"],"split_matches":SPLITS,"pre_feature_count":len(contract["pre_features"]),"live_feature_count":len(contract["live_features"]),"p1_feature_count":len(contract["p1_features"]),"python":platform.python_version()}
    (out/"run_config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8"); (out/"p2_summary.json").write_text(json.dumps(summary,indent=2,default=float),encoding="utf-8")
    print("P2 core evaluation complete; strict checklist/report finalization pending:",out)


if __name__ == "__main__":
    main()
