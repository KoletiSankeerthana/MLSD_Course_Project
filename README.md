# Employee Salary Prediction – End-to-End MLOps Pipeline

## Project Overview

This project demonstrates the implementation of a complete end-to-end MLOps pipeline for an Employee Salary Prediction application using modern MLOps tools and technologies. The system predicts whether an employee earns more than or less than 50K salary based on demographic and employment-related features from the Adult Income dataset.

The project covers the complete machine learning lifecycle including:

* Machine Learning Pipeline Development
* Data and Model Versioning using DVC and AWS S3
* Git and GitHub Version Control
* Docker Containerization
* GitHub Actions CI/CD Automation
* Kubernetes Deployment and Orchestration
* Streamlit-based Web Application Deployment

---

# Project Architecture

```text id="o8j2mv"
Dataset → Preprocessing → Model Training → DVC Tracking → AWS S3 Storage
       → GitHub Version Control → Docker Containerization
       → GitHub Actions CI/CD → Kubernetes Deployment
       → Streamlit Application Serving
```

---

# Technologies Used

| Technology     | Purpose                      |
| -------------- | ---------------------------- |
| Python         | Machine Learning Development |
| Scikit-learn   | Model Training               |
| Pandas         | Data Processing              |
| Streamlit      | Web Application              |
| Git & GitHub   | Version Control              |
| DVC            | Data & Model Versioning      |
| AWS S3         | Remote Artifact Storage      |
| Docker         | Containerization             |
| DockerHub      | Container Registry           |
| GitHub Actions | CI/CD Automation             |
| Kubernetes     | Container Orchestration      |

---

# Project Structure

```text id="r7p4qx"
MLSD_Course_Project/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── data/
│   └── adult.csv
│
├── models/
│   └── salary_model.pkl
│
├── src/
│   ├── preprocess.py
│   └── train.py
│
├── streamlit_app/
│   └── app.py
│
├── deployment.yaml
├── service.yaml
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Machine Learning Workflow

## 1. Data Preprocessing

The dataset is preprocessed using:

* Missing value handling
* Label Encoding
* Feature transformation
* Train-test splitting

---

## 2. Model Training

A Random Forest Classifier model is trained using Scikit-learn.

### Model Hyperparameters

```python id="n5m8tr"
RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)
```

---

## 3. Model Evaluation

The model is evaluated using:

* Accuracy Score
* Classification Report

---

# DVC and AWS S3 Integration

DVC was used to track:

* Dataset files
* Trained machine learning models

AWS S3 was configured as the remote storage backend for versioned artifacts.

### DVC Commands Used

```bash id="m2q7rv"
dvc init
dvc add data/adult.csv
dvc add models/salary_model.pkl
dvc remote add -d myremote s3://employee-salary-mlsd
dvc push
```

---

# Docker Containerization

The Streamlit application and machine learning model were containerized using Docker.

### Docker Build

```bash id="v8p3qx"
docker build -t employee-salary-app .
```

### Docker Run

```bash id="r4n1tw"
docker run -p 8501:8501 employee-salary-app
```

---

# DockerHub Integration

The Docker image was pushed to DockerHub for centralized container storage.

### Docker Push

```bash id="x7m5qp"
docker push <dockerhub-username>/employee-salary-app:latest
```

---

# GitHub Actions CI/CD

GitHub Actions was configured to automate:

* Dependency installation
* Docker image build validation
* Workflow execution on every push to GitHub

### CI/CD Trigger

Every Git push automatically triggers the workflow.

---

# Kubernetes Deployment

Kubernetes was used to deploy and manage the containerized application.

## Deployment Features

* Replica Management
* Pod Orchestration
* Service Exposure
* Scalable Deployment

### Kubernetes Commands

```bash id="k9n2rv"
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get pods
kubectl get services
```

---

# Streamlit Application Access

The application was accessed locally using Kubernetes port forwarding.

### Port Forwarding

```bash id="t5m8qx"
kubectl port-forward service/employee-salary-service 8501:8501
```

### Access URL

```text id="u3n7tr"
http://localhost:8501
```

---

# CI/CD Iteration Example

Hyperparameter tuning was performed by modifying:

* `n_estimators`
* `max_depth`

After updating `train.py`, GitHub Actions automatically triggered the CI/CD pipeline.

### Git Commands Used

```bash id="w6p4rv"
git add src/train.py
git commit -m "experiment: tune RF hyperparameters"
git push
```

---

# Key MLOps Concepts Demonstrated

* End-to-End MLOps Workflow
* Reproducible Machine Learning Pipelines
* Data and Model Version Control
* Cloud Storage Integration
* Docker-Based Containerization
* CI/CD Pipeline Automation
* Kubernetes Deployment and Scaling
* Streamlit Application Serving

---

# Conclusion

This project successfully demonstrates the implementation of a production-oriented MLOps pipeline for an Employee Salary Prediction system. The workflow integrates machine learning development, version control, cloud artifact management, containerization, CI/CD automation, and Kubernetes orchestration into a scalable and reproducible deployment pipeline.

The project provides hands-on practical implementation of modern MLOps concepts and deployment strategies used in real-world machine learning systems.
