from sklearn.metrics import classification_report, roc_curve, auc
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import KFold

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    predictions = model.predict(X_test)
    metrics = classification_report(y_test, predictions.round(), output_dict=True)
    
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    
    return metrics, fpr, tpr, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve using plotly"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                            name=f'ROC curve (AUC = {roc_auc:.2f})',
                            mode='lines'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                            name='Random',
                            mode='lines',
                            line=dict(dash='dash')))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    return fig

def cross_validate(model, X, y, n_splits=5):
    """Perform cross-validation"""
    kf = KFold(n_splits=n_splits, shuffle=True)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train, epochs=5, verbose=0)
        score = model.evaluate(X_val, y_val, verbose=0)
        scores.append(score[1])  # accuracy
        
    return np.mean(scores), np.std(scores)
