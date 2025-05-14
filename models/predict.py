def predict_next_day(model_path, last_window):
    """Прогноз цен с проверкой формы данных"""
    model = load_model(model_path)
    scaler = joblib.load("data/processed/scaler.pkl")
    
    # Преобразование и проверка данных
    last_window = np.array(last_window)
    if np.isnan(last_window).any():
        last_window = np.nan_to_num(last_window, nan=np.nanmean(last_window))
    
    # Правильное преобразование формы
    scaled_window = scaler.transform(last_window.reshape(-1, 1)).reshape(1, -1, 1)
    
    scaled_pred = model.predict(scaled_window)
    return scaler.inverse_transform(scaled_pred)[0]