# Параметры модели
MODEL_CONFIG = {
    "window_size": 30,          # Размер окна для LSTM
    "epochs": 50,               # Количество эпох обучения
    "batch_size": 32,
    "test_size": 0.2            # Доля тестовых данных
}

# Пути к данным
DATA_PATHS = {
    "raw_data": "data/raw/ecommerce_sales.csv",
    "processed_data": "data/processed/price_history.csv"
}