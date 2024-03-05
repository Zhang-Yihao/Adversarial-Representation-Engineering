import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from tqdm import tqdm


def classifier_trainer(model,
                       train_loader,
                       criterion=nn.CrossEntropyLoss(),
                       lr=1e-3,
                       eps=1e-3,
                       epochs=10,
                       device=None,
                       is_eval=True,
                       eval_dset=None,
                       eval_labels=None):
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)
    if device:
        model = model.to(device)
    else:
        model = model.cuda()

    for epoch in range(epochs):
        for i, (data, labels) in enumerate(train_loader):
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i + 1 == len(train_loader):
                    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    if is_eval:
        assert eval_dset is not None, "eval dataset is empty"
        assert eval_labels is not None, "eval labels is empty"
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            outputs = model(eval_dset)
            _, predicted = torch.max(outputs.data, 1)
            total = eval_labels.size(0)
            correct = (predicted == eval_labels).sum().item()

            print('Test Accuracy of the model on the test data: {} %'.format(100 * correct / total))


def peft_model_finetune(model,
                        train_dataloader,
                        eval_dataloader,
                        epochs=1,
                        lr=1e-3,
                        eps=1e-3,
                        criterion=nn.CrossEntropyLoss(),
                        device="cuda:0"):
    if not device:
        device = model.device
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)
    for epoch in range(epochs):
        print("Epoch:", epoch)
        model.train()
        train_loss = 0
        for xb_yb in tqdm(train_dataloader):
            xb, yb = xb_yb
            xb = xb.to(device)
            yb = yb.to(device)
            with autocast():
                outputs = model(xb)
                loss = criterion(outputs, yb)
                train_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        for xb_yb in tqdm(eval_dataloader):
            xb, yb = xb_yb
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                outputs = model(xb)
            loss = criterion(outputs, yb)
            eval_loss += loss.detach().float()

        eval_loss_total = (eval_loss / len(eval_dataloader)).item()
        train_loss_total = (train_loss / len(train_dataloader)).item()
        print(f"{epoch=:<2}  {train_loss_total=:.4f}  {eval_loss_total=:.4f}")
