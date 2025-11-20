# Security Controls (OWASP LLM Top 10 + MITRE ATLAS)

This repository now enforces concrete MLSecOps controls across Docker, Jenkins, Git/DVC artifacts, and MLflow:

1. **Pre-flight checks (`security/security_checks.py`)**
   - Runs inside Jenkins before any training stage.
   - Validates MLflow Tracking URI is local/isolated (OWASP LLM05 / ATLAS Initial Access).
   - Verifies dataset + SQLite artifacts via SHA-256 fingerprints to catch poisoning or tampering (OWASP LLM04, ML02, ATLAS Poisoning).
   - Emits an AI Bill of Materials (`security/ai_bom.json`) for audit trails mapped to ATLAS.

2. **Training-time integrity + tagging**
   - `sklearn/train_sklearn.py` and `autogluon/train_autogluon.py` abort if their local data/db hashes drift unexpectedly. Hash values are logged to MLflow parameters for traceability.
   - MLflow runs tag their compliance scope (`security_frameworks`, `owasp_llm_controls`) so downstream consumers can query/report on coverage.

3. **Jenkins / Docker hardening**
   - A new “Security - Preflight” stage runs inside a pinned Python 3.10 image, mounts the workspace read-only (except for BOM output), and archives evidence into Jenkins (traceable in Git/DVC history).
   - Training stages still run in isolated Docker containers, and the BOM artifacts are fingerprinted to ensure Git history matches deployed assets.

4. **Operational guidance**
   - `security/baseline_hashes.json` is the single source of truth for trusted data artifacts. Update it only after verifying new datasets with DVC/Git commit reviews.
   - Set `MLSECOPS_ALLOW_DATA_DRIFT=1` only when rotating datasets intentionally; the hash change will still be logged for auditing.
   - For remote MLflow tracking servers, explicitly acknowledge the risk by setting `MLFLOW_ALLOW_REMOTE=1` so Jenkins logs the exception.

These controls satisfy the immediate OWASP / MITRE requirements and serve as hooks for future layers (prompt firewalls, runtime guardrails, etc.).
