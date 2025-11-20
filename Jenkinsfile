pipeline {
  agent any
  options { timestamps() }

  parameters {
    choice(
      name: 'TARGET',
      choices: ['sklearn','autogluon','both'],
      description: 'Hangi deneyi çalıştırayım?'
    )
  }

  environment {
    // Jenkins container'larının bağlandığı güvenli MLflow endpoint'i
    MLFLOW_TRACKING_URI = 'http://host.docker.internal:5000'
    MLFLOW_UI_BASE      = 'http://host.docker.internal:5000'
    MLFLOW_ALLOW_REMOTE = '0'   // remote MLflow default olarak yasak

    // Hosttaki proje yolu (Mac Desktop)
    WORKDIR = '/Users/elifsena/Desktop/mlops-ci'
  }

  stages {

    stage('Smoke: Docker erişimi') {
      steps {
        sh 'docker version || true'
      }
    }

    stage('Security - Preflight') {
      agent {
        docker {
          image 'python:3.10-slim'
          args "-v ${WORKDIR}:/work -w /work"
        }
      }
      steps {
        sh """
          set -e
          python -m pip install -U pip
          pip install --no-cache-dir mlflow==2.14.1

          python security/security_checks.py \
            --target ${params.TARGET} \
            --mlflow-uri ${MLFLOW_TRACKING_URI} \
            --write-bom security/ai_bom.json
        """
      }
      post {
        always {
          archiveArtifacts artifacts: 'security/ai_bom.json', fingerprint: true
        }
      }
    }

    stage('Train - scikit-learn') {
      when {
        anyOf {
          expression { params.TARGET == 'sklearn' }
          expression { params.TARGET == 'both' }
        }
      }
      environment {
        MLFLOW_EXPERIMENT_NAME = 'exp_sklearn_secure'
      }
      agent {
        docker {
          image 'python:3.10-slim'
          args "-u root:root -v ${WORKDIR}:/work -w /work"
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
            --random_state 42 \
            | tee "$WORKSPACE/sklearn_train.log"

          cp mlflow_run_url.txt "$WORKSPACE/sklearn_run.txt"
        '''
      }
      post {
        always {
          archiveArtifacts artifacts: 'sklearn_train.log,sklearn_run.txt', fingerprint: true
        }
      }
    }

    stage('Train - AutoGluon') {
      when {
        anyOf {
          expression { params.TARGET == 'autogluon' }
          expression { params.TARGET == 'both' }
        }
      }
      environment {
        MLFLOW_EXPERIMENT_NAME = 'Advertising-AutoGluon-secure'
      }
      agent {
        docker {
          image 'python:3.10-slim'
          // ARM Mac için amd64 emülasyonu + root
          args "--platform linux/amd64 -u root:root -v ${WORKDIR}:/work -w /work"
        }
      }
      steps {
        sh '''
          set -e
          apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1

          python -m pip install -U pip
          pip install --no-cache-dir mlflow==2.14.1 autogluon.tabular==1.4.0

          cd /work/autogluon
          MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
          MLFLOW_EXPERIMENT_NAME=Advertising-AutoGluon-secure \
          python train_autogluon.py \
            --experiment Advertising-AutoGluon-secure \
            --time_limit 60 \
            | tee "$WORKSPACE/autogluon_train.log"

          cp mlflow_run_url.txt "$WORKSPACE/autogluon_run.txt"
        '''
      }
      post {
        always {
          archiveArtifacts artifacts: 'autogluon_train.log,autogluon_run.txt', fingerprint: true
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
        if (fileExists('autogluon_run.txt')) {
          echo "AUTOGLUON → " + readFile('autogluon_run.txt').trim()
        }
      }
    }
    failure {
      echo 'Build FAILED. Log: sklearn_train.log / autogluon_train.log'
    }
  }
}
