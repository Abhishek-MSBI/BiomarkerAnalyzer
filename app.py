import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
from src.data_processing import (load_genomic_data, get_sequence_stats,
                               process_sequences)
from src.model import build_cnn_model, train_model_with_progress, prepare_sequence_data
from src.evaluate import evaluate_model, plot_roc_curve, cross_validate
from src.visualizations import (plot_kmer_distribution, plot_sequence_length_distribution,
                               plot_training_history)

st.set_page_config(page_title="Biomarker Discovery Platform", layout="wide")

st.title("🧬 Viral Disease Biomarker Discovery Platform")

# Sidebar
st.sidebar.header("Parameters")
k_mer_size = st.sidebar.slider("K-mer Size", 4, 8, 6)
epochs = st.sidebar.slider("Training Epochs", 5, 50, 10)
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.1, 0.01, 0.001, 0.0001],
    value=0.001,
    format_func=lambda x: f"{x:.4f}"
)

# File upload
st.header("1. Data Upload and Processing")
uploaded_file = st.file_uploader("Upload FASTA file", type=['fasta', 'fa'])

if uploaded_file:
    # Load and process data
    sequences_df = load_genomic_data(uploaded_file)
    
    if sequences_df is not None:
        # Display basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sequence Statistics")
            stats = get_sequence_stats(sequences_df)
            for key, value in stats.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("Sequence Length Distribution")
            fig = plot_sequence_length_distribution(sequences_df)
            st.plotly_chart(fig)
        
        # Process k-mers
        st.header("2. K-mer Analysis")
        kmer_freq = process_sequences(sequences_df, k=k_mer_size)
        
        st.subheader("K-mer Distribution")
        fig = plot_kmer_distribution(kmer_freq)
        st.plotly_chart(fig)
        
        # Model training
        st.header("3. Model Training")
        
        if st.button("Train Model"):
            # Prepare data (using dummy labels for demonstration)
            X = prepare_sequence_data(kmer_freq, k=k_mer_size)
            y = np.random.randint(0, 2, size=len(sequences_df))
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Build and train model
            model = build_cnn_model((X.shape[1], 1), learning_rate)
            
            st.subheader("Training Progress")
            history = train_model_with_progress(model, X_train, y_train,
                                              X_test, y_test, epochs)
            
            # Plot training history
            st.subheader("Training History")
            fig = plot_training_history(history)
            st.plotly_chart(fig)
            
            # Model evaluation
            st.header("4. Model Evaluation")
            metrics, fpr, tpr, roc_auc = evaluate_model(model, X_test, y_test)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['Precision', 'Recall', 'F1-Score'],
                    'Value': [
                        metrics['1']['precision'],
                        metrics['1']['recall'],
                        metrics['1']['f1-score']
                    ]
                })
                st.table(metrics_df)
            
            with col2:
                st.subheader("ROC Curve")
                fig = plot_roc_curve(fpr, tpr, roc_auc)
                st.plotly_chart(fig)
            
            # Cross-validation
            st.subheader("Cross-validation Results")
            cv_mean, cv_std = cross_validate(model, X, y)
            st.write(f"Mean CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")

else:
    st.info("Please upload a FASTA file to begin analysis")

# Footer
st.markdown("---")
st.markdown("""
Built with Streamlit • TensorFlow • Biopython  
Created by ABHISHEK S R
""")