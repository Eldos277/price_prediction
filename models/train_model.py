from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from utils.data_loader import load_raw_data, create_price_history
from utils.preprocessor import prepare_lstm_data
from config import MODEL_CONFIG
import numpy as np

def build_model(input_shape):
    """Архитектура LSTM-модели"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(input_shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train():
    """Обучение и сохранение модели"""
    df = load_raw_data()
    price_data = create_price_history(df)
    
    X, y, scaler = prepare_lstm_data(price_data)
    split_idx = int(len(X) * (1 - MODEL_CONFIG["test_size"]))
    
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(
        X[:split_idx], y[:split_idx],
        epochs=MODEL_CONFIG["epochs"],
        batch_size=MODEL_CONFIG["batch_size"],
        validation_data=(X[split_idx:], y[split_idx:])
    )
    
    model.save("models/price_predictor.keras")
    return model, scaler