import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from preprocess import preprocess_pipeline

def train_model(data_path, model_save_path):
    """
    Trains an XGBoost Classifier on the preprocessed data, evaluates it,
    and saves the model.
    
    Parameters:
    data_path (str): Path to the raw CSV data.
    model_save_path (str): Path to save the serialized model dictionary.
    
    Returns:
    tuple: Trained model and a dict of evaluation metrics.
    """
    print(f"Starting model training pipeline using data from: {data_path}")
    
    # 1. Preprocess the data
    X_train, X_test, y_train, y_test, features = preprocess_pipeline(data_path)
    
    # 2. Handle class imbalance
    # Calculate scale_pos_weight to balance recall and precision for the rare positive class
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos
    print(f"Data imbalance check - Negatives: {num_neg}, Positives: {num_pos}")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")
    
    # 3. Initialize XGBoost Classifier with robust parameters
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # 4. Train the model
    print("Fitting XGBoost Classifier...")
    xgb_model.fit(X_train, y_train)
    
    # 5. Evaluate the model
    print("Evaluating model performance...")
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("\n" + "="*40)
    print("MODEL EVALUATION METRICS ON TEST SET:")
    print("="*40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("="*40 + "\n")
    
    # 6. Save the model package
    # Saving both the model, the training feature list, and evaluation metrics for reference
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    model_package = {
        'model': xgb_model,
        'features': features,
        'metrics': metrics
    }
    
    joblib.dump(model_package, model_save_path)
    print(f"Model package successfully saved to: {model_save_path}")
    
    return xgb_model, metrics

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "Data", "ai4i2020.csv")
    model_save_path = os.path.join(base_dir, "models", "xgb_model.pkl")
    
    try:
        train_model(data_path, model_save_path)
    except Exception as e:
        print(f"Training failed: {str(e)}")
