import pandas as pd
from config import DATA_PATHS

def load_raw_data():
    """Загрузка и обработка данных с правильными названиями столбцов"""
    try:
        # Чтение данных с указанием правильного столбца даты
        df = pd.read_csv(
            DATA_PATHS["raw_data"],
            parse_dates=['Order Date'],  # Используем правильное название столбца
            dayfirst=True  # Если даты в формате DD/MM/YYYY
        )
        
        # Переименование для совместимости с остальным кодом
        df = df.rename(columns={
            'Order Date': 'OrderDate',
            'Product Name': 'ProductName',
            'Unit Price': 'UnitPrice'
        })
        
        # Преобразование цены (на случай если есть символы валют)
        df['UnitPrice'] = df['UnitPrice'].astype(str).str.replace('[^\d.]', '', regex=True).astype(float)
        
        return df[['OrderDate', 'ProductName', 'UnitPrice']]
    
    except Exception as e:
        print(f"Ошибка загрузки данных: {str(e)}")
        raise

def create_price_history(df):
    """Создание временных рядов цен (остается без изменений)"""
    history = df.pivot_table(
        index='OrderDate',
        columns='ProductName',
        values='UnitPrice',
        aggfunc='mean'
    ).ffill().bfill()
    
    history.to_csv(DATA_PATHS["processed_data"])
    return history