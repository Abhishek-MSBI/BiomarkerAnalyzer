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
        showlegend=False
    )
    return fig

def plot_sequence_length_distribution(sequences_df):
    """Plot sequence length distribution"""
    lengths = sequences_df['sequence'].str.len()
    
    fig = px.histogram(lengths,
                      title='Sequence Length Distribution',
                      labels={'value': 'Sequence Length', 'count': 'Frequency'},
                      nbins=30)
    return fig

def plot_training_history(history):
    """Plot training history"""
    fig = go.Figure()
    
    # Add traces for accuracy
    fig.add_trace(go.Scatter(y=history['accuracy'],
                            name='Training Accuracy',
                            mode='lines'))
    fig.add_trace(go.Scatter(y=history['val_accuracy'],
                            name='Validation Accuracy',
                            mode='lines'))
    
    fig.update_layout(
        title='Model Training History',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        hovermode='x'
    )
    return fig
