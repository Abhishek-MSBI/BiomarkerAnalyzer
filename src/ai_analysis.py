import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd

def get_sequence_features(sequence):
    """Extract basic sequence features without deep learning"""
    features = []
    # Length
    features.append(len(sequence))
    # GC content
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    features.append(gc_content)
    # Basic composition
    for base in ['A', 'T', 'G', 'C']:
        features.append(sequence.count(base) / len(sequence))
    # Dinucleotide frequencies
    dinucleotides = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
                     'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
    for di in dinucleotides:
        count = sum(1 for i in range(len(sequence)-1) if sequence[i:i+2] == di)
        features.append(count / (len(sequence)-1))
    return np.array(features)

def analyze_sequences(sequences_df):
    """Analyze sequences using basic feature extraction"""
    try:
        # Extract features for each sequence
        features = []
        for seq in sequences_df['sequence']:
            features.append(get_sequence_features(seq))
        features = np.array(features)

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(features)

        # Perform clustering
        n_clusters = min(3, len(sequences_df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)

        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)

        analysis_results = {
            'embeddings': features,
            'reduced_embeddings': reduced_features,
            'similarity_matrix': similarity_matrix,
            'clusters': clusters,
            'feature_importance': calculate_feature_importance(features)
        }
        return analysis_results

    except Exception as e:
        st.error(f"Error in sequence analysis: {str(e)}")
        return None

def calculate_feature_importance(features):
    """Calculate feature importance based on variance"""
    feature_names = ['Length', 'GC Content'] + \
                   ['A Freq', 'T Freq', 'G Freq', 'C Freq'] + \
                   ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC',
                    'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']

    # Calculate variance of each feature
    importance = np.var(features, axis=0)
    # Normalize importance scores
    importance = importance / np.sum(importance)

    return dict(zip(feature_names, importance))