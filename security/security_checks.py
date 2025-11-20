"""
Security pre-flight checks for Jenkins/Docker/MLflow pipeline.

Implements lightweight OWASP LLM Top 10 + MITRE ATLAS controls:
 - Integrity validation for training artifacts (poisoning protection)
 - Tracking URI isolation enforcement (reduces prompt hijacking surface)
 - AI Bill of Materials (SBOM-like) export for audit / ATLAS mapping
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parents[1]
BASELINE_FILE = Path(__file__).with_name("baseline_hashes.json")
DEFAULT_BOM_PATH = Path(__file__).with_name("ai_bom.json")
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def _load_baseline() -> Dict[str, Dict[str, str]]:
    if not BASELINE_FILE.exists():
        raise SystemExit(f"Baseline hash dosyası bulunamadı: {BASELINE_FILE}")
    try:
        return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"baseline_hashes.json okunamadı: {exc}") from exc


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files(targets: List[str], allow_drift: bool = False) -> List[dict]:
    baseline = _load_baseline()
    report = []
    errors = []

    for component in targets:
        files = baseline.get(component, {})
        if not files:
            print(f"[warn] {component} için baseline hash bulunamadı.", file=sys.stderr)
        for rel, expected in files.items():
            path = BASE_DIR / rel
            if not path.exists():
                msg = f"{rel} bulunamadı"
                errors.append(msg)
                report.append({"component": component, "path": rel, "status": "missing"})
                continue
            digest = _sha256(path)
            status = "ok" if digest == expected else "mismatch"
            if digest != expected:
                msg = (
                    f"Integrity check FAILED for {rel}: "
                    f"expected {expected}, got {digest}"
                )
                errors.append(msg)
            report.append(
                {
                    "component": component,
                    "path": rel,
                    "status": status,
                    "expected_sha256": expected,
                    "observed_sha256": digest,
                }
            )
    if errors and not allow_drift:
        raise SystemExit("\n".join(errors))
    if errors and allow_drift:
        for msg in errors:
            print(f"[warn] {msg}", file=sys.stderr)
    return report


def enforce_tracking_uri(uri: str) -> None:
    parsed = urlparse(uri)
    host = (parsed.hostname or "").lower()
    scheme = (parsed.scheme or "").lower()
    if scheme in ("", "file") or host in LOCAL_HOSTS:
        return
    allow_remote = os.getenv("MLFLOW_ALLOW_REMOTE", "").lower() in {"1", "true"}
    if not allow_remote:
        raise SystemExit(
            "MLflow tracking URI lokal değil ve MLFLOW_ALLOW_REMOTE=1 değil: "
            f"{uri}"
        )
    print(
        "[warn] Remote MLflow URI kullanımına izin verildi (MLFLOW_ALLOW_REMOTE=1).",
        file=sys.stderr,
    )


def write_bom(report: List[dict], uri: str, dest: Path) -> None:
    bom = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "mlflow_tracking_uri": uri,
        "git_commit": os.getenv("GIT_COMMIT"),
        "components": report,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(bom, indent=2), encoding="utf-8")
    print(f"[info] AI BOM yazıldı: {dest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLSecOps güvenlik kontrolleri")
    parser.add_argument(
        "--target",
        choices=["sklearn", "autogluon", "both"],
        default="both",
        help="Hangi deneyin dosyaları valide edilsin?",
    )
    parser.add_argument(
        "--mlflow-uri",
        required=True,
        help="MLflow tracking URI (lokal değilse hata verir).",
    )
    parser.add_argument(
        "--allow-data-drift",
        action="store_true",
        help="Hash uyuşmazlıklarını fatal yerine uyarı olarak işaretler.",
    )
    parser.add_argument(
        "--write-bom",
        type=Path,
        default=DEFAULT_BOM_PATH,
        help="Üretilen güvenlik raporunun beklenen yolu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = (
        ["sklearn", "autogluon"]
        if args.target == "both"
        else [args.target]
    )
    enforce_tracking_uri(args.mlflow_uri)
    report = verify_files(targets, allow_drift=args.allow_data_drift)
    write_bom(report, args.mlflow_uri, args.write_bom)


if __name__ == "__main__":
    main()
