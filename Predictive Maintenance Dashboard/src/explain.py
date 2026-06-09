import os
import joblib
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
from typing import Tuple, List, Dict, Any


def _get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model_package(model_path: str = None):
    if model_path is None:
        model_path = os.path.join(_get_base_dir(), 'models', 'xgb_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def shap_explain(instance: pd.DataFrame, model_package: Dict[str, Any] = None) -> Tuple[List[Tuple[str, float]], Any]:
    """
    Compute SHAP values for a single instance or small DataFrame.

    Returns: (top_features, plotly_figure)
    - top_features: list of (feature, contribution)
    - plotly_figure: bar chart of absolute SHAP values
    """
    if model_package is None:
        model_package = load_model_package()

    model = model_package['model']
    features = model_package.get('features')

    # Prepare instance in expected column order
    X = instance.copy()
    # Ensure order
    if features is not None:
        X = X.reindex(columns=features, fill_value=0.0)

    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap_values may return list for multi-output; handle common binary classifier format
    if isinstance(shap_values, list):
        vals = shap_values[1]
    else:
        vals = shap_values

    # Handle different shapes: prefer per-instance signed SHAP values
    if vals.ndim == 2:
        # vals shape (n_samples, n_features)
        row = vals[0]
    else:
        row = vals

    feat_signed = list(zip(X.columns.tolist(), row.tolist()))
    # Sort by absolute contribution
    feat_sorted = sorted(feat_signed, key=lambda x: abs(x[1]), reverse=True)

    # Build signed horizontal bar chart for top features
    top_n = min(10, len(feat_sorted))
    top_feats = feat_sorted[:top_n]
    names = [f[0] for f in top_feats]
    vals_plot = [f[1] for f in top_feats]

    colors = ['#EF553B' if v > 0 else '#636EFA' for v in vals_plot]
    fig = go.Figure(go.Bar(x=vals_plot[::-1], y=names[::-1], orientation='h', marker_color=colors[::-1]))
    fig.update_layout(title='SHAP feature impacts (signed)', template='plotly_dark', height=420)

    return feat_sorted, fig


def get_prediction_explanation(input_df: pd.DataFrame, model_path: str = None) -> Dict[str, Any]:
    pkg = load_model_package(model_path)
    feats, fig = shap_explain(input_df, pkg)
    # return top 5 features including signed shap values and impact direction
    top_f = feats[:5]
    top_list = []
    for f, v in top_f:
        impact = 'increases_risk' if v > 0 else 'decreases_risk' if v < 0 else 'neutral'
        top_list.append({'feature': f, 'shap_value': float(v), 'impact': impact})

    return {
        'top_features': top_list,
        'plotly_figure': fig
    }


if __name__ == '__main__':
    # simple smoke example: requires model and valid features
    print('SHAP explain module loaded')
