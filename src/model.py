import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import numpy as np
import streamlit as st

def build_cnn_model(input_shape, learning_rate=0.001):
    """Build and compile CNN model"""
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
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
    feature_size = 4**k  # Size for k-mer frequencies
    X = np.zeros((1, feature_size, 1))

    for i, freq in enumerate(kmer_frequencies.values()):
        if i < feature_size:
            X[0, i, 0] = freq

    return X.astype(np.float32)  # Ensure float32 dtype for TensorFlow