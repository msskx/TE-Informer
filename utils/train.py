import torch
from torch import nn


def fit(epoch, model, trainloader, testloader, lr=0.003):
    total = 0
    running_loss = 0
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')

        x = x.to(torch.float32)
        y = y.to(torch.float32)
        y_pred = model(x)
        y_pred = y_pred.to(torch.float32)

        y = y.squeeze()
        y_pred = y_pred.squeeze()
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total += y.size(0)
            running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader.dataset)

    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')

            x = x.to(torch.float32)
            y = y.to(torch.float32)
            y_pred = model(x)
            y_pred = y_pred.to(torch.float32)
            y = y.squeeze()
            y_pred = y_pred.squeeze()

            loss = loss_fn(y_pred, y)
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)

    print('epoch: ', epoch,
          'loss_train: ', round(epoch_loss, 4),
          'test_loss: ', round(epoch_test_loss, 4),
          )

    return epoch_loss, epoch_test_loss
