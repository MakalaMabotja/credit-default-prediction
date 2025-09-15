# Credit Default Prediction

## 📌 Overview

This project was developed for the [Zindi African Credit Scoring Challenge](https://zindi.africa/competitions/african-credit-scoring-challenge/submissions), where the goal was to **predict the likelihood of a customer defaulting on a loan** using financial and demographic data.

The solution achieves an **F1 score of 0.8844**, placing it among the top-performing approaches. Beyond accuracy, the system was designed with a **microservices architecture** to ensure scalability, modularity, and real-world deployment readiness.

---

## 🏗 Project Architecture

The system follows a **microservices design**, with each service running independently and communicating via APIs. Azure cloud services are used for scalability, monitoring, and deployment.

### Microservices

1. **Data Ingestion & Processing**

   - Ingests CSV data, transforms, and loads into a SQL star schema.
   - Feature engineering (time intelligence, customer dimensions, interest rates).

2. **Data Analysis & Statistical Testing**

   - Performs EDA and statistical checks.
   - Ensures new data aligns with historical patterns.

3. **Model Training & Logging**

   - Trains LightGBM, XGBoost, Random Forest, and Decision Trees.
   - Cross-validation, hyperparameter tuning, and MLflow logging.

4. **Model Interpretation & Ensembling**

   - Provides explainability with SHAP/LIME.
   - Combines models via stacking and weighted averaging.

5. **Model Deployment & API Service**
   - Deploys models via FastAPI endpoints.
   - Serves predictions and a credit scoring function that bins probabilities into risk categories.

---

## 🔧 Tech Stack

- **Languages & Libraries**: Python (`pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `mlflow`, `shap`, `lime`)
- **Frameworks**: FastAPI, Pydantic
- **Database**: Azure SQL (Star Schema with SQLAlchemy)
- **Orchestration & Cloud**: Azure Data Factory, Azure Databricks, Azure ML, Azure App Service, Azure API Management
- **Deployment**: Docker, Azure Kubernetes Service (AKS)

---

## ⚙️ Workflow

1. **Ingestion** → CSV → Azure Blob Storage → SQL Database
2. **Analysis** → EDA + Statistical Tests → Logs anomalies
3. **Training** → ML models trained & logged with MLFlow
4. **Interpretation** → Explainability & ensembles
5. **Deployment** → REST API for predictions + credit scoring

---

## 🔒 Key Considerations

- **Scalability**: Independent microservices on AKS
- **Monitoring**: Azure Monitor, MLFlow, Application Insights
- **Security**: Azure API Management + Key Vault for secrets
- **CI/CD**: Automated pipelines via Azure DevOps
- **Data Governance**: Compliance with GDPR/POPIA

---

## 📊 Results

- Achieved **F1 Score: 0.8844** on Zindi challenge test data.
- Delivered a scalable **credit scoring function** for real-world financial decision-making.

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/MakalaMabotja/credit-default-prediction.git
   ```
