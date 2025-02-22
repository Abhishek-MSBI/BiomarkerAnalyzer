import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_kmer_distribution(kmer_freq):
    """Plot k-mer frequency distribution"""
    df = pd.DataFrame(list(kmer_freq.items()), columns=['k-mer', 'frequency'])
    df = df.sort_values('frequency', ascending=False).head(20)

    fig = px.bar(df,
                 x='k-mer',
                 y='frequency',
                 title='Top 20 k-mer Frequencies',
                 labels={'k-mer': 'K-mer Sequence', 'frequency': 'Frequency'})

    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, l=50, r=30, b=50)
    )
    return fig

def plot_sequence_length_distribution(sequences_df):
    """Plot sequence length distribution"""
    lengths = sequences_df['sequence'].str.len()

    fig = px.histogram(lengths,
                      title='Sequence Length Distribution',
                      labels={'value': 'Sequence Length', 'count': 'Frequency'},
                      nbins=30)

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, l=50, r=30, b=50)
    )
    return fig

def plot_training_history(history):
    """Plot training history"""
    fig = go.Figure()

    # Add traces for accuracy
    fig.add_trace(go.Scatter(y=history['accuracy'],
                            name='Training Accuracy',
                            mode='lines',
                            line=dict(color='#2ecc71')))
    fig.add_trace(go.Scatter(y=history['val_accuracy'],
                            name='Validation Accuracy',
                            mode='lines',
                            line=dict(color='#e74c3c')))

    fig.update_layout(
        title='Model Training History',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        hovermode='x',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, l=50, r=30, b=50)
    )
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    # Get feature importance from the model's first dense layer
    weights = model.layers[-2].get_weights()[0]
    importance = np.abs(weights).mean(axis=1)

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Feature': feature_names[:len(importance)],
        'Importance': importance
    })
    df = df.sort_values('Importance', ascending=False).head(20)

    fig = px.bar(df,
                 x='Feature',
                 y='Importance',
                 title='Top 20 Feature Importance',
                 labels={'Feature': 'K-mer', 'Importance': 'Importance Score'})

    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, l=50, r=30, b=50)
    )
    return fig