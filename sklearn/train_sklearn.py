import os
import json
import math
import sqlite3
import argparse
import warnings
import hashlib
from pathlib import Path
from urllib.parse import urlparse  # güvenlik etiketi için

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_URL = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
DB_PATH = "data.db"
TABLE = "california_housing"
Y_COL = "median_house_value"

# OWASP Top 10 / MITRE ATLAS uyumluluğu için veri bütünlüğü kontrolleri
DATA_BASELINE = {
    "data.csv": "bb9ca4041a66e0ed691cb825188f16fdab518435c4950a080951be2fdfe30d4d",
    "data.db": "51819365c569aa25e8367a62ba56537148a03dc6781034bd50dd8caab09e740b",
}
ALLOW_DRIFT_ENV = "MLSECOPS_ALLOW_DATA_DRIFT"

# UI adresini kodun içine gömmüyoruz; ortam değişkeninden alıyoruz
MLFLOW_UI_BASE = os.getenv("MLFLOW_UI_BASE", "")


def ensure_sqlite_exists():
    """
    DB yoksa CSV'yi indirip SQLite'a yaz. İNTERNET YOKSA SENTETİK VERİ ÜRETME!
    İnternet yok ve DB de yoksa hata ver.
    """
    if Path(DB_PATH).exists():
        return

    Path("data").mkdir(exist_ok=True)
    csv_local = Path("data.csv")
    try:
        import requests

        r = requests.get(CSV_URL, timeout=60)
        r.raise_for_status()
        csv_local.write_bytes(r.content)
        df = pd.read_csv(csv_local)
    except Exception as e:
        raise RuntimeError(
            f"Veri indirilemedi ve '{DB_PATH}' mevcut değil. "
            f"İnternet bağlantını ya da yerel CSV'yi kontrol et. Hata: {e}"
        )

    with sqlite3.connect(DB_PATH) as c:
        df.to_sql(TABLE, c, if_exists="replace", index=False)


def load_df():
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql_query(f"SELECT * FROM {TABLE}", c)


def _verify_local_artifacts():
    """
    Veri zehirlenmesini erkenden yakalamak için veri/db dosyalarının hash'lerini
    baseline ile karşılaştırır. Drift'e bilinçli olarak izin verilecekse
    MLSECOPS_ALLOW_DATA_DRIFT=1 set edilebilir.
    """
    allow_drift = os.getenv(ALLOW_DRIFT_ENV, "").lower() in {"1", "true"}
    results = {}
    for rel, expected in DATA_BASELINE.items():
        p = Path(__file__).resolve().parent / rel
        if not p.exists():
            raise RuntimeError(f"Gerekli dosya bulunamadı: {p}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        results[rel] = digest
        if digest != expected and not allow_drift:
            raise RuntimeError(
                f"{rel} integrity check failed (expected {expected}, got {digest})."
            )
        if digest != expected and allow_drift:
            print(
                f"[warn] {rel} hash değişti (expected {expected}, got {digest})",
                flush=True,
            )
    return results


def log_fig(fig, name: str):
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    p = out / name
    fig.tight_layout()
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(p))


def residual_plots(y_true, y_pred):
    resid = y_true - y_pred

    f1 = plt.figure()
    plt.scatter(y_true, y_pred, s=6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Prediction vs Actual")
    log_fig(f1, "pred_vs_actual.png")

    f2 = plt.figure()
    plt.scatter(y_pred, resid, s=6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    log_fig(f2, "residuals.png")

    f3 = plt.figure()
    plt.hist(resid, bins=40)
    plt.xlabel("Residual")
    plt.title("Residuals Histogram")
    log_fig(f3, "residuals_hist.png")


def coef_bar(model, feat):
    coefs = np.asarray(model.coef_)
    order = np.argsort(np.abs(coefs))[::-1]
    f = plt.figure(figsize=(10, 4))
    plt.bar([feat[i] for i in order], coefs[order])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("coef")
    plt.title("Coefficients")
    log_fig(f, "coefs.png")


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def _emit_mlflow_link(run, tracking_uri: str, exp_id: str):
    """
    MLflow run linkini mlflow_run_url.txt'ye yazar ve konsola basar.
    Öncelik MLFLOW_UI_BASE (http/https). O yoksa, tracking_uri http/https ise onu kullan.
    file: vs. durumlarda exp/run id basar.
    """
    try:
        run_id = run.info.run_id
        base = (MLFLOW_UI_BASE or "").strip()
        if not base or not base.startswith(("http://", "https://")):
            base = (tracking_uri or "").strip()
        if base.startswith(("http://", "https://")):
            url = f"{base}/#/experiments/{exp_id}/runs/{run_id}"
        else:
            url = f"exp_id={exp_id} run_id={run_id}"
        Path("mlflow_run_url.txt").write_text(url + "\n", encoding="utf-8")
        print("MLflow run:", url, flush=True)
    except Exception as e:
        print(f"[warn] mlflow_run_url.txt yazılamadı: {e}", flush=True)


def _tag_tracking_security(tracking_uri: str):
    """
    Tracking server'ın local/izole olup olmadığını run tag'i olarak işaretler.

    - localhost, 127.0.0.1 veya file: ise -> 'tracking_is_local_or_file' = True
    - Diğer durumlarda uyarı verip False yazar.
    """
    parsed = urlparse(tracking_uri)
    host = parsed.hostname or ""
    scheme = (parsed.scheme or "").lower()

    is_local = scheme in ("file", "") or host in ("127.0.0.1", "localhost")

    mlflow.set_tag("tracking_uri", tracking_uri)
    mlflow.set_tag("tracking_is_local_or_file", bool(is_local))

    if not is_local:
        print(
            f"[warn] Tracking URI '{tracking_uri}' localhost/file değil. "
            f"Ağ erişimini firewall / reverse proxy ile kısıtladığından emin ol.",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "exp_sklearn"),
    )
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    # Güvenli MLflow server'ı ortam değişkeninden alıyoruz (örn: http://127.0.0.1:5000)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(args.experiment)

    ensure_sqlite_exists()
    artifact_hashes = _verify_local_artifacts()
    df = load_df()
    X_cols = [c for c in df.columns if c != Y_COL]
    X = df[X_cols].values
    y = df[Y_COL].values
    kf = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    with mlflow.start_run(run_name="linear-regression") as run:
        mlflow.set_tag("jenkins_job", os.getenv("JOB_NAME"))
        mlflow.set_tag("jenkins_build_number", os.getenv("BUILD_NUMBER"))
        mlflow.set_tag("git_commit", os.getenv("GIT_COMMIT"))
        mlflow.set_tag("script_abs_path", str(Path(__file__).resolve()))
        try:
            mlflow.set_tag(
                "script_sha1",
                hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12],
            )
        except Exception:
            pass

        # --- Güvenlik / framework tag'leri (OWASP + MITRE ATLAS) ---
        tracking_uri = mlflow.get_tracking_uri()
        is_local = (
            tracking_uri.startswith("http://127.0.0.1")
            or tracking_uri.startswith("http://localhost")
            or tracking_uri.startswith("file:")
        )

        mlflow.set_tag("security_frameworks", "OWASP-LLM-Top10, MITRE-ATLAS")
        mlflow.set_tag("tracking_uri", tracking_uri)
        mlflow.set_tag("tracking_is_local_or_file", str(is_local))

        # Beklenmedik remote MLflow'a loglamayı reddet
        if not is_local:
            raise RuntimeError(
                f"MLflow tracking URI güvenli değil görünüyor: {tracking_uri}. "
                "Sadece local/file MLflow'a loglamaya izin veriyorum."
            )

        # HİÇBİR SECRET LOGLANMIYOR → sadece hiperparametre ve veri yolu bilgileri
        mlflow.log_param("features", json.dumps(X_cols))
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("sqlite_db", str(Path(DB_PATH).resolve()))
        mlflow.log_param("sqlite_table", TABLE)
        for rel, digest in artifact_hashes.items():
            mlflow.log_param(f"sha256::{rel}", digest)
        mlflow.set_tag("security_frameworks", "OWASP-LLM-Top10,MITRE-ATLAS")
        mlflow.set_tag("owasp_llm_controls", "LLM01,LLM02,LLM03,LLM05,LLM06,LLM10")

        # Model eğitimi
        model = LinearRegression()
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)

        # Metrikler
        mae = mean_absolute_error(yte, yp)
        r2 = r2_score(yte, yp)
        r = rmse(yte, yp)
        mlflow.log_metric("TEST_MAE", float(mae))
        mlflow.log_metric("TEST_RMSE", float(r))
        mlflow.log_metric("TEST_R2", float(r2))

        # Baseline dummy model
        d = DummyRegressor(strategy="mean").fit(Xtr, ytr)
        b_rmse = rmse(yte, d.predict(Xte))
        mlflow.log_metric("BASELINE_RMSE", float(b_rmse))

        # CV metrikleri
        rmse_cv = -cross_val_score(
            LinearRegression(), X, y, cv=kf, scoring="neg_root_mean_squared_error"
        )
        r2_cv = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
        mlflow.log_metric("CV_RMSE_MEAN", float(rmse_cv.mean()))
        mlflow.log_metric("CV_RMSE_STD", float(rmse_cv.std()))
        mlflow.log_metric("CV_R2_MEAN", float(r2_cv.mean()))
        mlflow.log_metric("CV_R2_STD", float(r2_cv.std()))

        # Şekiller ve artefact'lar
        residual_plots(yte, yp)
        coef_bar(model, X_cols)
        Path("artifacts/sqlite_info.txt").write_text(
            f"Rows={len(df)} | Table={TABLE}", encoding="utf-8"
        )
        mlflow.log_artifact("artifacts/sqlite_info.txt")

        # Model artefact
        mlflow.sklearn.log_model(model, "model", input_example=Xte[:3])

        # Run linkini üret ve kaydet
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        exp = client.get_experiment_by_name(args.experiment)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        _emit_mlflow_link(run, tracking_uri, exp.experiment_id)

        print(
            f">> TEST MAE={mae:.2f} | RMSE={r:.2f} | "
            f"R2={r2:.4f} | Baseline RMSE={b_rmse:.2f}"
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
