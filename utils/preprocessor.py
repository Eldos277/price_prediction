def prepare_lstm_data(data):
    """Подготовка данных для LSTM с обработкой NaN"""
    # Заполнение пропусков
    data = data.ffill().bfill()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))  # Явное указание формы
    
    joblib.dump(scaler, "data/processed/scaler.pkl")
    
    X, y = [], []
    window_size = MODEL_CONFIG["window_size"]
    
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i].reshape(window_size, 1))
        y.append(scaled_data[i])
    
    return np.array(X), np.array(y), scaler