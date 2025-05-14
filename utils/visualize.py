import matplotlib.pyplot as plt

def plot_predictions(actual, predicted, product_name):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Фактические цены')
    plt.plot(predicted, label='Прогноз')
    plt.title(f"Динамика цен: {product_name}")
    plt.legend()
    plt.savefig(f"results/{product_name}_forecast.png")
    plt.close()