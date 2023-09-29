import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from utils.plots import plot_prediction, plot_scatter
from utils import metrics


def evaluate(model=None, test_dl=None, model_name="LSTM", ptype='Single'):
    print("$" * 50)
    print("Evaluate Model")
    print("$" * 50)

    loss_fn = nn.MSELoss()
    test_running_loss = 0
    test_running_mae_loss = 0
    loss_mae = nn.L1Loss()
    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            print(f"观测特征值:{x.shape},预测真实值:{y.shape}")
            y_pred = model(x)
            y_pred = y_pred.to(torch.float32)
            y = y.squeeze()
            y_pred = y_pred.squeeze()
            print(f"真实值:{y.shape},预测值:{y_pred.shape}")
            print(f"真实值:{y.view(-1).shape},预测值:{y_pred.view(-1).shape}")

            fig, ax = plt.subplots(figsize=(12, 6), dpi=400)
            x_idx = range(1, len(y_pred) + 1)  # 初始化X轴下标
            if ptype == 'Single':
                src = y.detach().numpy()  # 真实值
                prediction = y_pred.detach().numpy()  # 预测值
            else:
                src = np.array([i[0].detach().numpy() for i in y_pred])
                prediction = np.array([i[0].detach().numpy() for i in y])
            ###############################################################

            ax.plot(x_idx, prediction, label="pre")
            ax.plot(x_idx, src, label="true")
            ax.set_title("Predicted value and true value")
            ax.set_xlabel('point-in-time')
            ax.set_ylabel('prediction for aqi')
            ax.legend()
            plt.show()
            ###############################################################
            # 保存预测值绘图
            # 散点图
            plot_scatter(true_value=src, predicted_value=prediction, model_name=model_name, ptype=ptype)
            ###############################################################
            plot_prediction(x_index=x_idx, src=src, prediction=prediction, model_name=model_name, ptype=ptype)
            ################################################################
            loss = loss_fn(y_pred, y)  # 使用Pytorch的MSE
            test_running_loss += loss.item()

            epoch_test_loss = test_running_loss / len(test_dl.dataset)
            # epoch_test_loss = test_running_loss

            #####################################
            loss2 = loss_mae(y_pred, y)
            test_running_mae_loss += loss2.item()
            epoch_test_mae_loss = test_running_mae_loss / len(test_dl.dataset)
            # epoch_test_mae_loss = test_running_mae_loss

            mae, mse, rmse, mape, mspe, r2 = metrics.metric(prediction, src)
            # mae, mse, rmse, mape, mspe, r2 = metrics.metric(y_pred.detach().numpy(),
            #                                                 y.detach().numpy())
            print(
                'test_mse_loss： ', round(epoch_test_loss, 4),
                'test_mae_loss： ', round(epoch_test_mae_loss, 4),
            )

            print(f"MAE:{mae},\n",
                  f"MSE:{mse},\n",
                  f"RMSE:{rmse}\n",
                  f"MAPE:{mape}\n",
                  f"MSPE:{mspe}\n",
                  f"R2:{r2}\n",
                  )
            break
