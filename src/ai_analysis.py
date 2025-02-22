import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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
        kmeans = KMeans(n_clusters=min(3, len(sequences_df)), random_state=42)
        clusters = kmeans.fit_predict(features)

        analysis_results = {
            'embeddings': features,
            'similarity_matrix': similarity_matrix,
            'clusters': clusters
        }
        return analysis_results

    except Exception as e:
        st.error(f"Error in sequence analysis: {str(e)}")
        return None

def compute_similarity_matrix(features):
    """Compute similarity matrix between sequences"""
    try:
        similarity_matrix = cosine_similarity(features)
        return similarity_matrix
    except Exception as e:
        st.error(f"Error computing similarity matrix: {str(e)}")
        return None