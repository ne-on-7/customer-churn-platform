# 📉 Customer Churn Intelligence Platform

A full-stack ML platform that predicts customer churn, explains predictions with SHAP, and serves results through a REST API and interactive React dashboard with an AI-powered chat assistant.

Built on the **IBM Telco Customer Churn dataset** (7,043 customers) — from exploratory analysis through model training, explainability, and deployment.

> **5 models** · **5-fold stratified CV** · **ROC-AUC 0.8468** · **904% ROI** · **SHAP explainability** · **A/B testing** · **AI Chat**

---

## ✨ Features

**🤖 AI Chat Assistant**
- Natural language interface powered by Claude API with tool use
- Ask questions like "What's the churn risk for a senior citizen on month-to-month?" and get real-time predictions
- SSE streaming for responsive, real-time answers

**🔬 ML Pipeline**
- 5-model comparison: Logistic Regression, Random Forest, XGBoost, Gradient Boosting, PyTorch Neural Network
- 5-fold stratified cross-validation with class balancing
- 7 business-driven engineered features (tenure buckets, spend trends, contract risk scores)

**🔍 Explainability**
- SHAP-based global feature importance and per-customer explanations
- Top 3 churn reasons returned with every prediction

**🧪 Experimentation**
- A/B testing framework with 4 intervention types (discount, email, upgrade, loyalty program)
- Power analysis, statistical significance testing, and segment-level breakdowns

**📊 Dashboard**
- React + Tailwind CSS frontend with dark mode
- 9 views: AI Chat, Predict, Models, Explorer, Impact, A/B Testing, Watchlist, History, Batch Upload
- Customer risk watchlist, batch CSV predictions, and prediction audit log

**🚀 Deployment**
- FastAPI REST API with interactive docs
- Dockerized multi-stage build (Node + Python)

---

## 📁 Project Structure

```
customer-churn-platform/
├── frontend/                     # React + Tailwind + Vite
│   ├── src/
│   │   ├── components/           # Chat, Predict, Models, Explorer, etc.
│   │   ├── lib/api.ts            # API client
│   │   └── types/index.ts        # TypeScript interfaces
│   └── vite.config.ts            # Proxy to FastAPI dev server
├── api/main.py                   # FastAPI REST API + AI chat
├── src/
│   ├── data_processing.py        # Load, clean, encode, split
│   ├── feature_engineering.py    # Business-driven features
│   ├── models.py                 # 5 model definitions (incl. PyTorch)
│   ├── train.py                  # Training pipeline with CV
│   ├── evaluate.py               # Metrics, plots, business impact
│   ├── explain.py                # SHAP explanations
│   └── experimentation.py        # A/B testing framework
├── models/                       # Trained model artifacts
├── plots/                        # Generated visualizations
├── data/
│   ├── raw/                      # IBM Telco dataset
│   └── processed/                # Train/test splits
├── notebooks/01_eda.ipynb        # Exploratory data analysis
├── Dockerfile
├── run.sh
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. Train all models

```bash
python -m src.train
```

### 3. Generate evaluation plots

```bash
python -m src.evaluate
```

### 4. Set up AI chat (optional)

```bash
export ANTHROPIC_API_KEY=your-key-here
```

### 5. Start the development servers

**Terminal 1 — Backend:**
```bash
uvicorn api.main:app --reload
```

**Terminal 2 — Frontend:**
```bash
cd frontend && npm run dev
```

Open `http://localhost:5173`

### 🐳 Docker

```bash
docker build -t churn-platform .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your-key churn-platform
```

Open `http://localhost:8000`

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict churn probability + SHAP explanations |
| `GET` | `/models` | Performance summary of all trained models |
| `GET` | `/health` | Health check |
| `GET` | `/data/overview` | Dataset statistics |
| `GET` | `/data/churn-rates?group_by=Contract` | Churn rates by segment |
| `GET` | `/data/distribution?feature=tenure` | Feature distribution histogram data |
| `POST` | `/business-impact` | Revenue impact calculator |
| `GET` | `/customers/high-risk?limit=20` | Top churn-risk customers |
| `POST` | `/predict/batch` | Bulk CSV prediction |
| `GET` | `/predictions/history` | Prediction audit log |
| `POST` | `/chat` | AI chat with SSE streaming |
| `POST` | `/experiments` | Create & run an A/B test experiment |
| `GET` | `/experiments` | List all experiments |
| `GET` | `/experiments/{id}` | Get experiment details & results |
| `POST` | `/experiments/power-analysis` | Power analysis calculator |
| `GET` | `/plots/{filename}` | Serve evaluation plot images |

---

## 📊 Dashboard

The React dashboard (`http://localhost:5173` in dev, `http://localhost:8000` in production) has 9 views:

- 💬 **AI Chat** — Ask natural language questions about churn, powered by Claude with tool use
- 🎯 **Predict** — Input customer details, get real-time churn probability with risk tier and SHAP-based reasons
- 📈 **Models** — Side-by-side metrics, ROC curves, PR curves for all 5 models
- 🗂️ **Explorer** — Dataset statistics, churn rates by segment, feature distributions
- 💼 **Impact** — Revenue impact calculator with confusion matrix breakdown
- 🧪 **A/B Testing** — Create experiments, power analysis, results visualization, experiment history
- 👀 **Watchlist** — Top-N highest churn risk customers with scores and recommended actions
- 📜 **History** — Audit log of all predictions made
- 📤 **Batch** — Upload CSV for bulk predictions with downloadable results

---

## 🧠 Model Performance

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| **Logistic Regression** | **0.8468** | ⭐ Best performer |
| Random Forest | 0.8347 | |
| Gradient Boosting | 0.8277 | |
| XGBoost | 0.8276 | |
| Neural Network (PyTorch) | 0.8176 | |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React, TypeScript, Tailwind CSS, Vite |
| Charts | Plotly.js (react-plotly.js) |
| Icons | Lucide React |
| AI Chat | Claude API (Anthropic) |
| Backend | FastAPI, Uvicorn |
| ML Models | scikit-learn, XGBoost, PyTorch |
| Explainability | SHAP |
| Data | pandas, NumPy |
| Containerization | Docker (multi-stage) |

---

## 👤 Author

**Pratheek Annadanam**
