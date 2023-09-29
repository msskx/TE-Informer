import numpy as np
import torch
import argparse
from data.data_loader import MYDataset
from models.lstm import LSTM
from models.mlp import MLP
from models.rnn import RNN
from models.tcn import TCNs
from models.former import TransformerModel
from utils.eval import evaluate
from utils.train import fit

#
# parser = argparse.ArgumentParser(description="AQI TimeSeries Prediction")
# parser.add_argument('--seq_length', type=int, default=30, help='Serial observations')
# parser.add_argument('--delay', type=int, default=1, help='Serial predictions')
# args = parser.parse_args()
np.random.seed(42)
if __name__ == '__main__':
    # Load data
    train_ds = MYDataset(is_train=True, seq_length=30, delay=1)
    test_ds = MYDataset(is_train=False, seq_length=30, delay=1)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128)
    print(next(iter(train_dl))[0].shape, next(iter(train_dl))[1].shape)
    input_size = next(iter(train_dl))[0].shape[-1]
    features_size = next(iter(train_dl))[0].shape[-2]
    output_size = next(iter(train_dl))[1].shape[-1]
    hidden_size = 64
    print(input_size, features_size, output_size)
    print(input_size * features_size)
    # Create Model
    # model = RNN(input_size=input_size, output_size=output_size)
    # model = TCNs(input_size=90, output_size=30, num_channels=[16, 32, 64], kernel_size=3, dropout=0, is_sequence=False)
    # model = MLP(features_size=90, input_size=11, hidden_size=64, output_size=30)
    model = TransformerModel(input_size=11, d_model=512, output_size=1, seq_len=30)
    print(model)
    epochs = 300
    lr = 0.0003
    train_loss = []
    test_loss = []
    # Train
    for epoch in range(epochs):
        epoch_loss, epoch_test_loss = fit(epoch,
                                          model,
                                          train_dl,
                                          test_dl,
                                          lr=lr)
        train_loss.append(epoch_loss)
        test_loss.append(epoch_test_loss)
    # Test
    evaluate(model=model, test_dl=test_dl, model_name="Informer", ptype='Single')
