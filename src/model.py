import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import streamlit as st

def build_cnn_model(input_shape, learning_rate=0.001):
    """Build and compile an enhanced CNN model"""
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(256, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

def train_model_with_progress(model, X_train, y_train, X_val, y_val, epochs=10):
    """Train model with progress bar and callbacks"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.empty()

    # Initialize history dictionary
    history = {
        'loss': [], 'accuracy': [], 'auc': [],
        'val_loss': [], 'val_accuracy': [], 'val_auc': []
    }

    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    for epoch in range(epochs):
        results = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1,
            verbose=0,
            callbacks=[early_stopping]
        )

        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)

        # Update metrics
        for metric in results.history:
            history[metric].append(results.history[metric][0])

        status_text.text(f'Epoch {epoch + 1}/{epochs}')
        metrics_container.write(
            f'Loss: {history["loss"][-1]:.4f}, '
            f'Accuracy: {history["accuracy"][-1]:.4f}, '
            f'AUC: {history["auc"][-1]:.4f}'
        )

        # Check early stopping
        if early_stopping.stopped_epoch:
            st.info(f"Training stopped early at epoch {epoch + 1}")
            break

    return history

def prepare_sequence_data(sequences_df, kmer_freq, k=6):
    """Prepare sequence data for model input"""
    num_sequences = len(sequences_df)
    feature_size = 4**k

    # Initialize feature matrix
    X = np.zeros((num_sequences, feature_size, 1))

    # Generate features for each sequence
    for i, sequence in enumerate(sequences_df['sequence']):
        # Count k-mers in this sequence
        sequence_kmers = {}
        for j in range(len(sequence) - k + 1):
            kmer = sequence[j:j+k]
            sequence_kmers[kmer] = sequence_kmers.get(kmer, 0) + 1

        # Fill the feature matrix
        for kmer, freq in sequence_kmers.items():
            if kmer in kmer_freq:
                idx = list(kmer_freq.keys()).index(kmer)
                if idx < feature_size:
                    X[i, idx, 0] = freq

    # Generate dummy labels for demonstration
    y = np.random.randint(0, 2, size=num_sequences)

    return X.astype(np.float32), y

def get_feature_importance(model, feature_names):
    """Extract feature importance from the model"""
    # Get weights from the last dense layer
    weights = np.abs(model.layers[-2].get_weights()[0])
    importance = np.mean(weights, axis=1)

    # Create feature importance dictionary
    feature_importance = {}
    for i, name in enumerate(feature_names):
        if i < len(importance):
            feature_importance[name] = float(importance[i])

    return feature_importance

def show_feature_importance(feature_importance):
    st.write("Feature Importance")
    st.bar_chart(feature_importance)