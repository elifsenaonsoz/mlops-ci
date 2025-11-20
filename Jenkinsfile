pipeline {
  agent any
  options { timestamps() }

  environment {
    // Güvenli MLflow endpoint'in (Mac Docker için host.docker.internal)
    MLFLOW_TRACKING_URI = 'http://host.docker.internal:5000'
    MLFLOW_UI_BASE      = 'http://host.docker.internal:5000'
    MLFLOW_ALLOW_REMOTE = '0'
  }

  stages {

    stage('Security - Preflight') {
      agent {
        docker {
          image 'python:3.10-slim'
          args "-v ${WORKSPACE}:/work -w /work"
        }
      }
      steps {
        sh '''
          set -e
          python -m pip install -U pip
          pip install --no-cache-dir mlflow==2.14.1

          cd /work
          python security/security_checks.py \
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
      agent {
        docker {
          image 'python:3.10-slim'
          args "-v ${WORKSPACE}:/work -w /work"
        }
      }
      steps {
        sh '''
          set -e
          cd /work
          python -m pip install -U pip
          pip install --no-cache-dir dvc

          dvc pull
          dvc status
        '''
      }
    }

    stage('Train - scikit-learn') {
      environment {
        MLFLOW_EXPERIMENT_NAME = 'exp_sklearn_secure'
      }
      agent {
        docker {
          image 'python:3.10-slim'
          args "-u root:root -v ${WORKSPACE}:/work -w /work"
        }
      }
      steps {
        sh '''
          set -e
          python -m pip install -U pip
          if [ -f /work/requirements_sklearn.txt ]; then
            pip install --no-cache-dir -r /work/requirements_sklearn.txt
          else
            pip install --no-cache-dir mlflow==2.14.1 scikit-learn pandas numpy matplotlib sqlalchemy requests
          fi

          cd /work/sklearn
          MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
          MLFLOW_EXPERIMENT_NAME=exp_sklearn_secure \
          python train_sklearn.py \
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

    // İstersen otomatik Git + DVC push için bu stage'i sonra ekleriz.
    // Şimdilik pipeline'ın sorunsuz yeşil olması için burayı boş bırakıyoruz.
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
