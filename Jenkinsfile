pipeline {
  agent any
  options { timestamps() }

  environment {
    // Sistemdeki Python yorumlayıcısı
    PY = 'python3'

    // MLflow server'ın local çalışıyor
    MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
    MLFLOW_UI_BASE      = 'http://127.0.0.1:5000'
    MLFLOW_ALLOW_REMOTE = '0'
  }

  stages {

    stage('Security - Preflight') {
      steps {
        sh '''
          set -e
          $PY -m pip install -U pip
          $PY -m pip install --no-cache-dir mlflow==2.14.1

          $PY security/security_checks.py \
            --target sklearn \
            --mlflow-uri ${MLFLOW_TRACKING_URI} \
            --write-bom security/ai_bom.json
        '''
      }
      post {
        always {
          archiveArtifacts artifacts: 'security/ai_bom.json', fingerprint: true
        }
      }
    }

    stage('Data Versioning (DVC pull)') {
      steps {
        sh '''
          set -e
          $PY -m pip install -U pip
          $PY -m pip install --no-cache-dir dvc

          dvc pull
          dvc status
        '''
      }
    }

    stage('Train - scikit-learn') {
      environment {
        MLFLOW_EXPERIMENT_NAME = 'exp_sklearn_secure'
      }
      steps {
        sh '''
          set -e
          $PY -m pip install -U pip
          if [ -f requirements_sklearn.txt ]; then
            $PY -m pip install --no-cache-dir -r requirements_sklearn.txt
          else
            $PY -m pip install --no-cache-dir mlflow==2.14.1 scikit-learn pandas numpy matplotlib sqlalchemy requests
          fi

          cd sklearn
          MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
          MLFLOW_EXPERIMENT_NAME=exp_sklearn_secure \
          $PY train_sklearn.py \
            --experiment exp_sklearn_secure \
            --test_size 0.2 \
            --random_state 42 | tee "$WORKSPACE/sklearn_train.log"

          cp mlflow_run_url.txt "$WORKSPACE/sklearn_run.txt"
        '''
      }
      post {
        always {
          archiveArtifacts artifacts: 'sklearn_train.log,sklearn_run.txt', fingerprint: true
        }
      }
    }
  }

  post {
    success {
      script {
        if (fileExists('sklearn_run.txt')) {
          echo "SKLEARN  → "  + readFile('sklearn_run.txt').trim()
        }
      }
    }
    failure {
      echo 'Build FAILED. Log için: sklearn_train.log'
    }
  }
}
