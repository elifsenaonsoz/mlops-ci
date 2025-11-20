#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoGluon Multimodal (Text + Tabular) + MLflow tracking
- Classification (0/1) veya Regression (1–5) desteği
- data/new/*.csv eklenirse run-time'da veriyi genişletir ve yeni run üretir
- M1/M2/M3 Mac uyumlu (GPU kapalı)
"""

import os, json, time, glob, argparse, hashlib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn  # hocanın istediği sklearn entegrasyon import'u
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score,
)

from autogluon.multimodal import MultiModalPredictor


# -------------------------------
# Utils
# -------------------------------
def seed_everything(seed: int = 42):
    np.random.seed(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def sha1_of_dataframe(df: pd.DataFrame) -> str:
    # veri parmak izi (MLflow'ta kanıt)
    b = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha1(b).hexdigest()[:12]


# -------------------------------
# Demo dataset (text + tabular)
# -------------------------------
def make_toy_dataset(task: str = "classification", n: int = 450, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos_words = ["mükemmel", "harika", "çok iyi", "beğendim", "şahane", "tatmin edici"]
    neg_words = ["kötü", "berbat", "hiç sevmedim", "vasat", "hayal kırıklığı", "yetersiz"]

    price = rng.normal(100, 20, n).clip(10, 300).round(2)
    length = rng.integers(5, 50, n)  # text length proxy
    helpful = (price/50 + rng.normal(0, 1, n)).clip(0, 10).round(2)

    reviews, labels = [], []
    for i in range(n):
        if rng.random() < 0.5:
            w = rng.choice(pos_words); labels.append(1)
        else:
            w = rng.choice(neg_words); labels.append(0)
        reviews.append(f"Ürün {w}. Fiyatı {price[i]:.0f} TL. Kullanımı {'kolay' if rng.random()<0.6 else 'zor'}.")

    df = pd.DataFrame({
        "review": reviews,
        "price": price,
        "length": length,
        "helpful_votes": helpful,
    })
    if task == "classification":
        df["label"] = labels
    else:
        base = 3 + 1.2*np.array(labels) + 0.01*(200-price)/10 + rng.normal(0, 0.4, n)
        df["stars"] = np.clip(base, 1, 5).round(2)
    return df


# -------------------------------
# New data normalizer (esnek)
# -------------------------------
def normalize_new_data(df: pd.DataFrame, task: str) -> pd.DataFrame:
    # kolonları normalize et (TR/EN alias'ları)
    df = df.rename(columns={c: c.lower().strip() for c in df.columns})

    aliases = {
        "review": ["review", "yorum", "text", "metin"],
        "price": ["price", "fiyat"],
        "length": ["length", "len", "uzunluk"],
        "helpful_votes": ["helpful_votes", "helpful", "oy", "yardim_oyu", "yardim_oy"],
    }
    target_aliases = {
        "classification": ["label", "target", "y", "etiket"],
        "regression": ["stars", "target", "y", "puan", "skor"],
    }

    def pick(name, pool):
        for a in pool:
            if a in df.columns: return a
        return None

    col_review = pick("review", aliases["review"])
    if not col_review:
        raise ValueError("Yeni veride en az 'review' kolonu bulunmalı.")
    out = pd.DataFrame()
    out["review"] = df[col_review].astype(str)

    col_price = pick("price", aliases["price"])
    out["price"] = pd.to_numeric(df[col_price], errors="coerce") if col_price else np.nan

    col_length = pick("length", aliases["length"])
    if col_length:
        out["length"] = pd.to_numeric(df[col_length], errors="coerce").fillna(0).astype(int)
    else:
        out["length"] = out["review"].str.split().str.len().fillna(0).astype(int)

    col_help = pick("helpful_votes", aliases["helpful_votes"])
    out["helpful_votes"] = pd.to_numeric(df[col_help], errors="coerce").fillna(0.0) if col_help else 0.0

    tgt_list = target_aliases["classification" if task=="classification" else "regression"]
    tgt_name = pick("target", tgt_list)
    if not tgt_name:
        raise ValueError("Yeni veride hedef kolon (label/stars) bulunmalı.")

    if task == "classification":
        out["label"] = pd.to_numeric(df[tgt_name], errors="coerce").fillna(0).astype(int).clip(0,1)
    else:
        out["stars"] = pd.to_numeric(df[tgt_name], errors="coerce")

    # eksik price'lar için median doldur
    if out["price"].isna().any():
        med = out["price"].median()
        if pd.isna(med): med = 0.0
        out["price"] = out["price"].fillna(med)

    cols = ["review","price","length","helpful_votes","label"] if task=="classification" \
           else ["review","price","length","helpful_votes","stars"]
    return out[cols]


# -------------------------------
# Plot helpers (log to MLflow)
# -------------------------------
def plot_and_log_classification(y_true, y_prob, run_dir):
    ensure_dir(run_dir)
    y_pred = (y_prob >= 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.xlabel("Pred"); plt.ylabel("True")
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    p1 = os.path.join(run_dir, "confusion_matrix.png")
    fig.savefig(p1, bbox_inches="tight"); plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(fpr, tpr); plt.plot([0,1],[0,1], "--")
    plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR")
    p2 = os.path.join(run_dir, "roc_curve.png")
    fig.savefig(p2, bbox_inches="tight"); plt.close(fig)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(rec, prec)
    plt.title("Precision-Recall Curve"); plt.xlabel("Recall"); plt.ylabel("Precision")
    p3 = os.path.join(run_dir, "pr_curve.png")
    fig.savefig(p3, bbox_inches="tight"); plt.close(fig)

    mlflow.log_artifact(p1, artifact_path="figures")
    mlflow.log_artifact(p2, artifact_path="figures")
    mlflow.log_artifact(p3, artifact_path="figures")


def plot_and_log_regression(y_true, y_pred, run_dir):
    ensure_dir(run_dir)
    # y vs yhat
    fig = plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    plt.xlabel("True"); plt.ylabel("Pred"); plt.title("Prediction vs Truth")
    p1 = os.path.join(run_dir, "pred_vs_true.png")
    fig.savefig(p1, bbox_inches="tight"); plt.close(fig)

    # residuals
    resid = y_true - y_pred
    fig = plt.figure()
    plt.hist(resid, bins=30)
    plt.xlabel("y - yhat"); plt.title("Residuals")
    p2 = os.path.join(run_dir, "residuals_hist.png")
    fig.savefig(p2, bbox_inches="tight"); plt.close(fig)

    mlflow.log_artifact(p1, artifact_path="figures")
    mlflow.log_artifact(p2, artifact_path="figures")


# -------------------------------
# Train function
# -------------------------------
def train_mm(
    task="classification",
    time_limit=120,
    seed=42,
    use_new_data=True,
    save_dir="outputs",
    presets="medium_quality",
    text_checkpoint="prajjwal1/bert-tiny",
):
    seed_everything(seed)
    ensure_dir(save_dir)

    # base dataset
    if task == "classification":
        df_base = make_toy_dataset(task="classification", n=450, seed=seed)
        label_col = "label"
    else:
        df_base = make_toy_dataset(task="regression", n=450, seed=seed)
        label_col = "stars"

    base_rows = len(df_base)
    base_sha1 = sha1_of_dataframe(df_base)

    # add new CSVs (runtime update)
    added_files, added_rows = [], 0
    df = df_base.copy()
    if use_new_data:
        for p in glob.glob("data/new/*.csv"):
            try:
                extra_raw = pd.read_csv(p)
                extra = normalize_new_data(extra_raw, task)
                df = pd.concat([df, extra], ignore_index=True)
                added_files.append(os.path.basename(p))
            except Exception as e:
                print(f"[WARN] {p} okunamadı / atlandı: {e}")

    added_rows = len(df) - base_rows
    final_sha1 = sha1_of_dataframe(df)

    # split
    strat = df[label_col] if (task=="classification") else None
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=strat)

    # MLflow
    exp = os.getenv("MLFLOW_EXPERIMENT_NAME", "autogluon-mm")
    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name=f"{task}-mm") as run:
        run_id = run.info.run_id
        mlflow.log_params({
            "task": task,
            "time_limit": time_limit,
            "seed": seed,
            "presets": presets,
            "text_checkpoint": text_checkpoint,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "base_data_sha1": base_sha1,
            "final_data_sha1": final_sha1,
            "added_files": ",".join(added_files) if added_files else "none",
            "added_rows": int(added_rows),
        })

        # Predictor
        predictor = MultiModalPredictor(
            label=label_col,
            problem_type=("binary" if task=="classification" else "regression"),
            path=os.path.join(save_dir, f"{task}_predictor_{int(time.time())}")
        )
        hyperparams = {
            "model.hf_text.checkpoint_name": text_checkpoint,
            "env.num_gpus": 0,  # Mac'te GPU'yu kapat
        }

        t0 = time.time()
        predictor.fit(
            train_data=train_df,
            time_limit=time_limit,
            presets=presets,
            hyperparameters=hyperparams,
            seed=seed,
        )
        mlflow.log_metric("train_seconds", time.time() - t0)

        # Evaluate
        figs_dir = os.path.join(save_dir, "figs")
        if task == "classification":
            proba = predictor.predict_proba(test_df)
            if hasattr(proba, "values"):
                # DataFrame olabilir: pozitif sınıf (1) kolonu
                if hasattr(proba, "columns"):
                    pos_col = 1 if (1 in getattr(proba, "columns", [])) else proba.columns[-1]
                    y_prob = proba[pos_col].values
                else:
                    y_prob = proba.values
            else:
                y_prob = np.asarray(proba)
            y_true = test_df[label_col].values.astype(int)
            y_pred = (y_prob >= 0.5).astype(int)

            acc = float(accuracy_score(y_true, y_pred))
            f1  = float(f1_score(y_true, y_pred))
            try:
                auc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                auc = float("nan")
            mlflow.log_metrics({"accuracy": acc, "f1": f1, "auc": auc})

            # report + figures
            rpt = classification_report(y_true, y_pred, output_dict=True)
            rpt_path = os.path.join(save_dir, "classification_report.json")
            with open(rpt_path, "w", encoding="utf-8") as f:
                json.dump(rpt, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(rpt_path, artifact_path="reports")

            plot_and_log_classification(y_true, y_prob, figs_dir)

        else:
            preds = predictor.predict(test_df)
            y_pred = preds.values if hasattr(preds, "values") else np.asarray(preds)
            y_true = test_df[label_col].values.astype(float)

            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(mean_squared_error(y_true, y_pred, squared=False))
            r2 = float(r2_score(y_true, y_pred))
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

            plot_and_log_regression(y_true, y_pred, figs_dir)

        # fit summary
        try:
            summary = predictor.fit_summary()
        except Exception as e:
            summary = {"fit_summary": f"unavailable: {e}"}
        summ_path = os.path.join(save_dir, "fit_summary.json")
        with open(summ_path, "w", encoding="utf-8") as f:
            json.dump(summary if isinstance(summary, dict) else {"summary": str(summary)}, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(summ_path, artifact_path="reports")

        # save model (AG 1.4.0: path paramı zorunlu)
        model_dir = predictor.path
        predictor.save(model_dir)
        mlflow.log_artifacts(model_dir, artifact_path="autogluon_model")

        # snapshot data
        tr_csv = os.path.join(save_dir, "train_snapshot.csv")
        te_csv = os.path.join(save_dir, "test_snapshot.csv")
        train_df.to_csv(tr_csv, index=False)
        test_df.to_csv(te_csv, index=False)
        mlflow.log_artifact(tr_csv, artifact_path="data")
        mlflow.log_artifact(te_csv, artifact_path="data")

        print(f"[OK] MLflow run_id={run_id}")
        print(f"🏃 View run at: {os.getenv('MLFLOW_TRACKING_URI','http://localhost:5000')}/#/experiments")

    return run_id


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoGluon Multimodal (text+tabular) + MLflow")
    parser.add_argument("--task", choices=["classification","regression"], default="classification")
    parser.add_argument("--time_limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_new_data", action="store_true", help="data/new/ içeriğini ekleme")
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--text_checkpoint", default="prajjwal1/bert-tiny")
    args = parser.parse_args()

    run_id = train_mm(
        task=args.task,
        time_limit=args.time_limit,
        seed=args.seed,
        use_new_data=not args.no_new_data,
        presets=args.presets,
        text_checkpoint=args.text_checkpoint,
    )
    print("Run tamamlandı:", run_id)
