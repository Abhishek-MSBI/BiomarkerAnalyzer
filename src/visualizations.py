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
    """Plot detailed training history"""
    fig = go.Figure()
    metrics = ['accuracy', 'loss', 'auc']
    colors = {'train': '#2ecc71', 'val': '#e74c3c'}

    for metric in metrics:
        # Training metrics
        if metric in history:
            fig.add_trace(go.Scatter(
                y=history[metric],
                name=f'Training {metric.capitalize()}',
                mode='lines',
                line=dict(color=colors['train'])
            ))

        # Validation metrics
        val_metric = f'val_{metric}'
        if val_metric in history:
            fig.add_trace(go.Scatter(
                y=history[val_metric],
                name=f'Validation {metric.capitalize()}',
                mode='lines',
                line=dict(color=colors['val'])
            ))

    fig.update_layout(
        title='Model Training History',
        xaxis_title='Epoch',
        yaxis_title='Metric Value',
        hovermode='x',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, l=50, r=30, b=50)
    )
    return fig

def plot_feature_importance(feature_importance, top_n=20):
    """Plot feature importance analysis"""
    df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    df = df.sort_values('Importance', ascending=False).head(top_n)

    fig = px.bar(df,
                 x='Feature',
                 y='Importance',
                 title=f'Top {top_n} Feature Importance',
                 labels={'Feature': 'K-mer', 'Importance': 'Importance Score'})

    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, l=50, r=30, b=50)
    )
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred.round())

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Negative', 'Positive'],
        y=['Negative', 'Positive'],
        colorscale='RdBu',
        showscale=True
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def plot_sequence_composition(sequences_df):
    """Plot sequence composition analysis"""
    base_counts = {'A': [], 'T': [], 'G': [], 'C': []}

    for seq in sequences_df['sequence']:
        total = len(seq)
        for base in base_counts:
            percentage = (seq.count(base) / total) * 100
            base_counts[base].append(percentage)

    fig = go.Figure()
    for base, percentages in base_counts.items():
        fig.add_trace(go.Box(
            y=percentages,
            name=base,
            boxpoints='outliers'
        ))

    fig.update_layout(
        title='Nucleotide Composition Distribution',
        yaxis_title='Percentage (%)',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def plot_sequence_similarity_matrix(similarity_matrix):
    """Plot sequence similarity matrix"""
    if similarity_matrix is None:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        colorscale='Viridis',
        showscale=True
    ))

    fig.update_layout(
        title='Sequence Similarity Matrix',
        xaxis_title='Sequence Index',
        yaxis_title='Sequence Index',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def plot_sequence_clusters(embeddings, clusters):
    """Plot sequence clusters using dimensionality reduction"""
    if embeddings is None or clusters is None:
        return None

    from sklearn.decomposition import PCA

    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'PC1': reduced_embeddings[:, 0],
        'PC2': reduced_embeddings[:, 1],
        'Cluster': clusters
    })

    fig = px.scatter(
        df, x='PC1', y='PC2',
        color='Cluster',
        title='Sequence Clusters',
        labels={'PC1': 'First Principal Component',
                'PC2': 'Second Principal Component',
                'Cluster': 'Cluster'}
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig