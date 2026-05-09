# Employee Salary Prediction - MLOps Project

This project predicts whether an employee's salary exceeds $50,000 based on demographic and employment data. It is structured as a production-ready MLOps project for ML System Design.

## 📁 Project Structure

```text
Employee_Salary_Prediction/
│
├── data/               # Raw and processed datasets
├── models/             # Trained models and encoders (Pickle files)
├── src/                # Modular ML pipeline source code
│   ├── preprocess.py   # Data cleaning and encoding
│   ├── train.py        # Model training script
│   └── predict.py      # Inference logic
├── streamlit_app/      # Deployment interface
│   └── app.py          # Streamlit dashboard
├── artifacts/          # Execution logs and plots
├── .github/workflows/  # CI/CD pipelines
├── requirements.txt    # Project dependencies
├── Dockerfile          # Containerization config
├── deployment.yaml     # Kubernetes Deployment
├── service.yaml        # Kubernetes Service
└── .gitignore          # Version control exclusions
```

## 🚀 How to Run

### 1. Local Development
First, install the dependencies:
```bash
pip install -r requirements.txt
```

#### Step A: Train the Model
This will clean the data and save the model to the `models/` folder.
```bash
python src/train.py
```

#### Step B: Run the Streamlit App
Launch the web interface for predictions:
```bash
streamlit run streamlit_app/app.py
```

### 2. Docker Deployment
Build the container:
```bash
docker build -t salary-app .
```
Run the container:
```bash
docker run -p 8501:8501 salary-app
```

### 3. Kubernetes Deployment
Ensure you have a cluster running (Minikube/Kind), then:
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## 📸 Screenshots to Take for Submission

For your project submission, capture the following:
1.  **Project Structure**: Screenshot of your file explorer showing the new modular folders.
2.  **Training Success**: The terminal output after running `python src/train.py` showing accuracy.
3.  **Streamlit Dashboard**: The web UI with inputs filled in.
4.  **Prediction Result**: The specific "Success" or "Warning" box showing the predicted salary.
5.  **Docker Build**: The terminal showing a successful `docker build` command.

---
Developed for ML System Design Course.
