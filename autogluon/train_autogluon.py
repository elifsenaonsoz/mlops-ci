import os
import sys
import sqlite3
import logging
import argparse
import hashlib

import requests
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from mlflow.tracking import MlflowClient
from autogluon.tabular import TabularPredictor
from urllib.parse import urlparse  # güvenlik etiketi için

# Konfigürasyonları kodun içine gömmek yerine ortam değişkenlerinden alıyoruz
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_UI_BASE = os.getenv("MLFLOW_UI_BASE", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Advertising-LR")

TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
SEED = int(os.getenv("SEED", "42"))

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "data.db")
CSV_PATH = os.path.join(DATA_DIR, "Advertising.csv")
CSV_URL = (
    "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/"
    "master/data/Advertising.csv"
)
DATA_BASELINE = {
    "data/Advertising.csv": "98350ef2035798df4dce301c8bbaf288030e678b43fa9713d5849d73cedc4cfd",
    "data/data.db": "2cdefd50be23efddd141120b97c1ddb7cbeba1d16dbe97436b4dcdef8981b41f",
}
ALLOW_DRIFT_ENV = "MLSECOPS_ALLOW_DATA_DRIFT"


def setup_logging(log_path="autogluon_train.log"):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    try:
        from autogluon.common.utils.log_utils import set_logger_verbosity

        set_logger_verbosity(3)
    except Exception as e:
        logging.warning(f"AutoGluon logger set edilemedi: {e}")


def ensure_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        r = requests.get(CSV_URL, timeout=30)
        r.raise_for_status()
        open(CSV_PATH, "wb").write(r.content)
    df = pd.read_csv(CSV_PATH)
    if df.columns[0].lower() in ("", "unnamed: 0", "index"):
        df = df.drop(columns=[df.columns[0]])
    eng = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("advertising", eng, if_exists="replace", index=False)
    return df


def load_from_sql():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT TV, Radio, Newspaper, Sales FROM advertising", con
    )
    con.close()
    return df


def verify_local_artifacts():
    allow_drift = os.getenv(ALLOW_DRIFT_ENV, "").lower() in {"1", "true"}
    results = {}
    for rel, expected in DATA_BASELINE.items():
        path = Path(__file__).resolve().parent / rel
        if not path.exists():
            raise RuntimeError(f"Gerekli dosya bulunamadı: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        results[rel] = digest
        if digest != expected and not allow_drift:
            raise RuntimeError(
                f"{rel} integrity check failed (expected {expected}, got {digest})."
            )
        if digest != expected and allow_drift:
            logging.warning(
                "%s hash değişti (expected %s, got %s)",
                rel,
                expected,
                digest,
            )
    return results


def plots(y_true, y_pred, run_dir):
    os.makedirs(run_dir, exist_ok=True)

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Predicted vs Actual")
    p1 = os.path.join(run_dir, "pred_vs_actual.png")
    plt.savefig(p1, bbox_inches="tight")
    plt.close()

    res = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, res, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Sales")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    p2 = os.path.join(run_dir, "residuals.png")
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    return [p1, p2]


def _emit_mlflow_link(run):
    try:
        exp_id = run.info.experiment_id
        run_id = run.info.run_id
        base = (MLFLOW_UI_BASE or "").strip()
        if not base or not base.startswith(("http://", "https://")):
            base = (MLFLOW_TRACKING_URI or "").strip()
        if base.startswith(("http://", "https://")):
            url = f"{base}/#/experiments/{exp_id}/runs/{run_id}"
        else:
            url = f"exp_id={exp_id} run_id={run_id}"
        with open("mlflow_run_url.txt", "w", encoding="utf-8") as f:
            f.write(url + "\n")
        logging.info(f"MLflow run: {url}")
    except Exception as e:
        logging.warning(f"mlflow_run_url.txt yazılamadı: {e}")


def _tag_tracking_security(tracking_uri: str):
    """
    Tracking server'ın local/izole olup olmadığını run tag'i olarak işaretler.
    """
    parsed = urlparse(tracking_uri)
    host = parsed.hostname or ""
    scheme = (parsed.scheme or "").lower()

    is_local = scheme in ("file", "") or host in ("127.0.0.1", "localhost")

    mlflow.set_tag("tracking_uri", tracking_uri)
    mlflow.set_tag("tracking_is_local_or_file", bool(is_local))

    if not is_local:
        logging.warning(
            "Tracking URI '%s' localhost/file değil. "
            "Ağ erişimini firewall / reverse proxy ile kısıtladığından emin ol.",
            tracking_uri,
        )


def main():
    setup_logging()
    logging.info("Job başladı. Ortam hazırlanıyor...")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name override",
    )
    ap.add_argument(
        "--time_limit",
        type=int,
        default=int(os.getenv("TIME_LIMIT", "60")),
    )
    args = ap.parse_args()
    exp_name = args.experiment or EXPERIMENT_NAME
    time_limit = int(args.time_limit)

    # Güvenli tracking server ortamdan geliyor (127.0.0.1:5000 vb.)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(exp_name)

    ensure_data()
    artifact_hashes = verify_local_artifacts()
    df = load_from_sql()
    logging.info(f"Veri hazır: {df.shape[0]} satır, {list(df.columns)}")

    idx_tr, idx_te = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=SEED
    )
    df_train = df.loc[idx_tr].copy()
    df_test = df.loc[idx_te].copy()
    logging.info(f"Train/Test: {len(df_train)}/{len(df_test)}")

    with mlflow.start_run() as run:
        # Jenkins / Git meta-tag'leri (güvenli, secret yok)
        mlflow.set_tag("jenkins_job", os.getenv("JOB_NAME"))
        mlflow.set_tag("jenkins_build_number", os.getenv("BUILD_NUMBER"))
        mlflow.set_tag("git_commit", os.getenv("GIT_COMMIT"))

        # Tracking URI güvenlik etiketi
        _tag_tracking_security(mlflow.get_tracking_uri())

        # Sadece hiperparametre / veri yolu loglanıyor, SECRET YOK
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("sqlite_db", os.path.abspath(DB_PATH))
        mlflow.log_param("sqlite_table", "advertising")
        mlflow.log_param("features", "TV,Radio,Newspaper")
        mlflow.log_param("target", "Sales")
        mlflow.log_param("autogluon_time_limit", time_limit)
        for rel, digest in artifact_hashes.items():
            mlflow.log_param(f"sha256::{rel}", digest)
        mlflow.set_tag("security_frameworks", "OWASP-LLM-Top10,MITRE-ATLAS")
        mlflow.set_tag("owasp_llm_controls", "LLM01,LLM02,LLM03,LLM05,LLM06,LLM10")

        save_dir = os.path.join("autogluon_out", f"ag-{run.info.run_id[:8]}")
        os.makedirs(save_dir, exist_ok=True)

        logging.info("AutoGluon fit() başlıyor...")
        predictor = TabularPredictor(
            label="Sales",
            problem_type="regression",
            eval_metric="rmse",
            path=save_dir,
        )
        # random_seed argümanı yeni sürümde desteklenmediği için kaldırıldı
        predictor.fit(
            train_data=df_train,
            time_limit=time_limit,
            presets="medium_quality_faster_train",
        )
        logging.info("fit() bitti, tahmin ve metrikler hesaplanıyor...")

        y_true = df_test["Sales"].values
        y_pred = predictor.predict(df_test.drop(columns=["Sales"])).values
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))
        logging.info(
            "Metrics -> RMSE=%.4f, MAE=%.4f, R2=%.4f", rmse, mae, r2
        )

        perf = predictor.evaluate(df_test, silent=True)
        for k, v in perf.items():
            try:
                mlflow.log_metric(f"ag_{k}", float(v))
            except Exception:
                pass

        lb = predictor.leaderboard(df_test, silent=True)
        lb_path = os.path.join(save_dir, "leaderboard.csv")
        lb.to_csv(lb_path, index=False)
        mlflow.log_artifact(lb_path, artifact_path="autogluon")

        for p in plots(y_true, y_pred, run_dir="artifacts"):
            mlflow.log_artifact(p, artifact_path="plots")

        from shutil import make_archive

        zip_path = make_archive(save_dir, "zip", root_dir=save_dir)
        mlflow.log_artifact(zip_path, artifact_path="autogluon_model")

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/sqlite_info.txt", "w", encoding="utf-8") as f:
            f.write(
                f"DB: {os.path.abspath(DB_PATH)}\n"
                f"TABLE: advertising\n"
                f"ROWS(train,test): {len(df_train)},{len(df_test)}\n"
            )
        mlflow.log_artifact("artifacts/sqlite_info.txt", artifact_path="data")

        _emit_mlflow_link(run)
        logging.info("Job başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
