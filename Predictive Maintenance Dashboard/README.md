# Predictive Maintenance Dashboard

Industry-grade Predictive Maintenance project built with Python, XGBoost, SHAP and Streamlit.

Overview
- Predict machine failures using AI4I 2020 dataset.
- Real-time simulator feeds the dashboard.
- SHAP explainability and automated maintenance recommendations.

Project structure

predictive-maintenance/
├── data/ (place `ai4i2020.csv` here)
├── models/ (trained model saved as `xgb_model.pkl`)
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── explain.py
│   └── simulator.py
├── dashboard/
│   └── dashboard.py
├── requirements.txt
└── README.md

Quickstart

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Place the dataset at `Data/ai4i2020.csv`.

3. Train the model:

```bash
python src/train.py
```

4. Run the dashboard (from project root):

```bash
streamlit run dashboard/dashboard.py
```

Notes
- The dashboard includes a simulator to generate sensor data; you can start it from the sidebar.
- Model is saved to `models/xgb_model.pkl` by `src/train.py`.
- SHAP explanations are computed on-demand; SHAP may take a couple seconds to compute.

License
- MIT-like for demo/educational use.
