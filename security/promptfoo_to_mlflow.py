import json
import mlflow
from pathlib import Path

report_path = Path("security/promptfoo_report.json")

def main():
    mlflow.set_experiment("exp_promptfoo_security")

    with mlflow.start_run(run_name="promptfoo-audit"):
        mlflow.set_tag("security_tool", "promptfoo")
        mlflow.set_tag("mlsecops_category", "red-team-test")

        with open(report_path, "r") as f:
            data = json.load(f)

        passed = sum(1 for t in data["tests"] if t["pass"])
        failed = sum(1 for t in data["tests"] if not t["pass"])

        mlflow.log_metric("tests_passed", passed)
        mlflow.log_metric("tests_failed", failed)
        mlflow.log_artifact(str(report_path), artifact_path="promptfoo")

if __name__ == "__main__":
    main()

