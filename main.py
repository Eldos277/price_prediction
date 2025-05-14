import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from models.train_model import train
from models.predict import predict_next_day
from utils.visualize import plot_predictions
from utils.data_loader import load_raw_data
import numpy as np

def main():
    # 1. Обучение модели
    model, scaler = train()
    
    # 2. Пример прогноза
    df = load_raw_data()
    price_data = df.pivot_table(index='OrderDate', columns='ProductName', values='UnitPrice')
    last_window = price_data.iloc[-30:].values  # Последние 30 дней
    
    prediction = predict_next_day("models/price_predictor.h5", last_window)
    print("Прогноз цен на следующий день:", prediction)
    
    # 3. Визуализация для первого товара
    product = price_data.columns[0]
    actual = price_data[product].values[-50:]
    predicted = [predict_next_day("models/price_predictor.h5", price_data[product].values[i-30:i])[0] 
                for i in range(30, 50)]
    
    plot_predictions(actual, predicted, product)

if __name__ == "__main__":
    main()