import os
import threading
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE = None


def _get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(model_path: str = None):
    global _MODEL_CACHE
    if model_path is None:
        model_path = os.path.join(_get_base_dir(), "models", "xgb_model.pkl")

    with _MODEL_LOCK:
        if _MODEL_CACHE is None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            _MODEL_CACHE = joblib.load(model_path)
        return _MODEL_CACHE


def _encode_type(type_val):
    mapping = {'L': 0, 'M': 1, 'H': 2}
    if pd.isna(type_val):
        return -1
    return mapping.get(type_val, -1)


def _build_feature_row(input_data: Dict[str, Any], features: list):
    # Expected input keys: Type, Air_temperature_K, Process_temperature_K,
    # Rotational_speed_rpm, Torque_Nm, Tool_wear_min
    d = {}
    Type = input_data.get('Type')
    air = float(input_data.get('Air_temperature_K'))
    proc = float(input_data.get('Process_temperature_K'))
    rpm = float(input_data.get('Rotational_speed_rpm'))
    torque = float(input_data.get('Torque_Nm'))
    tool = float(input_data.get('Tool_wear_min'))

    temp_diff = proc - air
    power = rpm * torque
    stress = torque * tool

    # populate possible columns
    candidates = {
        'Type': _encode_type(Type),
        'Air_temperature_K': air,
        'Process_temperature_K': proc,
        'Rotational_speed_rpm': rpm,
        'Torque_Nm': torque,
        'Tool_wear_min': tool,
        'temp_diff': temp_diff,
        'power': power,
        'stress': stress
    }

    for f in features:
        # if feature exists in candidates, use it; otherwise try lowercased variant
        if f in candidates:
            d[f] = candidates[f]
        else:
            # fallbacks for slightly different column namings
            key = f
            d[f] = candidates.get(key, 0.0)

    return pd.DataFrame([d])


def _status_from_percent(pct: float) -> str:
    if pct < 50:
        return 'Healthy'
    if pct < 80:
        return 'Warning'
    return 'Critical'


def get_recommendations(feat_row: Dict[str, Any]) -> list:
    recs = []
    try:
        proc = float(feat_row.get('Process_temperature_K', 0))
        air = float(feat_row.get('Air_temperature_K', 0))
        torque = float(feat_row.get('Torque_Nm', 0))
        tool = float(feat_row.get('Tool_wear_min', 0))
        stress = torque * tool

        if (proc - air) > 5:
            recs.append('Inspect cooling system (high temperature differential)')
        if torque > 70:
            recs.append('Inspect motor load and drive train (high torque)')
        if tool > 200:
            recs.append('Replace tool or schedule tooling maintenance (high tool wear)')
        if stress > 10000:
            recs.append('Check bearings and mechanical stress points')
    except Exception:
        pass
    if not recs:
        recs.append('Routine inspection recommended')
    return recs


def predict_failure(Type: str,
                    Air_temperature_K: float,
                    Process_temperature_K: float,
                    Rotational_speed_rpm: float,
                    Torque_Nm: float,
                    Tool_wear_min: float,
                    model_path: str = None) -> Dict[str, Any]:
    """
    Predict failure risk and provide status + recommendations.

    Returns dict: {risk_percent, status, prediction, recommendations}
    """
    package = load_model(model_path)
    model = package['model']
    features = package.get('features')

    inp = {
        'Type': Type,
        'Air_temperature_K': Air_temperature_K,
        'Process_temperature_K': Process_temperature_K,
        'Rotational_speed_rpm': Rotational_speed_rpm,
        'Torque_Nm': Torque_Nm,
        'Tool_wear_min': Tool_wear_min
    }

    X = _build_feature_row(inp, features)

    # Predict
    proba = float(model.predict_proba(X)[:, 1][0])
    risk_percent = round(proba * 100.0, 2)
    prediction = int(model.predict(X)[0])
    status = _status_from_percent(risk_percent)
    recs = get_recommendations(inp)

    return {
        'risk_percent': risk_percent,
        'status': status,
        'prediction': prediction,
        'recommendations': recs
    }


if __name__ == '__main__':
    # quick smoke test
    try:
        res = predict_failure('M', 295.0, 305.0, 1500, 40.0, 50.0)
        print(res)
    except Exception as e:
        print('Error during prediction:', e)
