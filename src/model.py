import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import numpy as np
import streamlit as st

def build_cnn_model(input_shape, learning_rate=0.001):
    """Build and compile CNN model"""
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

def train_model_with_progress(model, X_train, y_train, X_val, y_val, epochs=10):
    """Train model with progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.empty()
    
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(epochs):
        results = model.fit(X_train, y_train,
                          validation_data=(X_val, y_val),
                          epochs=1,
                          verbose=0)
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        
        # Update metrics
        history['loss'].append(results.history['loss'][0])
        history['accuracy'].append(results.history['accuracy'][0])
        history['val_loss'].append(results.history['val_loss'][0])
        history['val_accuracy'].append(results.history['val_accuracy'][0])
        
        status_text.text(f'Epoch {epoch + 1}/{epochs}')
        metrics_container.write(f'Loss: {history["loss"][-1]:.4f}, Accuracy: {history["accuracy"][-1]:.4f}')
    
    return history

def prepare_sequence_data(kmer_frequencies, k=6):
    """Prepare sequence data for model input"""
    X = np.array(list(kmer_frequencies.values())).reshape(-1, 4**k, 1)
    return X
