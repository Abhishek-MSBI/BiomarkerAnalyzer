import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import streamlit as st

def get_sequence_embeddings(sequences, model_name="facebook/esm2_t6_8M_UR50D"):
    """Get sequence embeddings using a pre-trained protein language model"""
    try:
        # Load tokenizer and model
        @st.cache_resource
        def load_model():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            return tokenizer, model

        tokenizer, model = load_model()

        # Process sequences in batches
        embeddings = []
        with torch.no_grad():
            for seq in sequences:
                inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
                outputs = model(**inputs)
                # Use mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding.numpy().flatten())

        return np.array(embeddings)
    except Exception as e:
        st.error(f"Error in sequence analysis: {str(e)}")
        return None

def analyze_sequences(sequences_df):
    """Analyze sequences using the pre-trained model"""
    try:
        # Get embeddings
        embeddings = get_sequence_embeddings(sequences_df['sequence'])
        if embeddings is None:
            return None

        # Perform analysis
        analysis_results = {
            'embeddings': embeddings,
            'similarity_matrix': compute_similarity_matrix(embeddings),
            'clusters': perform_clustering(embeddings)
        }
        return analysis_results
    except Exception as e:
        st.error(f"Error in sequence analysis: {str(e)}")
        return None

def compute_similarity_matrix(embeddings):
    """Compute similarity matrix between sequences"""
    try:
        # Normalize embeddings
        normalized = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        # Compute cosine similarity
        similarity_matrix = np.dot(normalized, normalized.T)
        return similarity_matrix
    except Exception as e:
        st.error(f"Error computing similarity matrix: {str(e)}")
        return None

def perform_clustering(embeddings, n_clusters=3):
    """Perform clustering on sequence embeddings"""
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        return clusters
    except Exception as e:
        st.error(f"Error in clustering: {str(e)}")
        return None
