import numpy as np
from matplotlib import pyplot as plt


def plot_prediction(x_index, src, prediction, model_name, ptype):
    """
    :param x_index: 绘图的X轴下标
    :param src: 真实值
    :param prediction:预测值
    :param model_name: 模型名称
    :param ptype: 预测类型单步 OR 多步
    :return: None
    """
    plt.figure(figsize=(15, 6))
    plt.rcParams.update({"font.size": 16})
    # plotting
    plt.plot(x_index, src, 'o--', color='limegreen', label='True', linewidth=2)
    plt.plot(x_index, prediction, 'o-', color='blue', label='Prediction', linewidth=2)
    #     plt.plot(x_index, prediction,'--', color = 'limegreen', label = 'Forecast', linewidth=2)

    # formatting
    plt.grid(visible=True, which='major', linestyle='solid')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', linestyle='dashed', alpha=0.5)

    plt.xlabel("Time step")
    plt.ylabel("AQI")
    plt.legend()
    plt.title(f"{model_name}_{ptype}_Prediction")
    # save
    plt.savefig(f"./img/{model_name}_{ptype}_Prediction.png")
    plt.close()


def plot_scatter(true_value, predicted_value,model_name, ptype):
    plt.figure(figsize=(6, 6))
    plt.scatter(true_value, predicted_value, c='tab:orange', alpha=0.5, edgecolors='none')

    plt.yscale('log')
    plt.xscale('log')
    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.title(f"{model_name}_{ptype}_Scatter")
    plt.savefig(f"./img/{model_name}_{ptype}_Scatter.png")
    plt.show()
