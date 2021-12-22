import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

EPOCHS = 20


class ChessDataset(Dataset):
    def __init__(self):
        self.data = np.load(Path("processed/dataset.npz"))
        self.X = self.data["X"]
        self.Y = self.data["Y"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # input layer
        self.l1 = nn.Linear(69, 100)  # input shape (69,) -> output to layer 100 units
        self.relu1 = nn.ReLU()

        # hidden layer 1
        self.l2 = nn.Linear(100, 100)  # input shape 100 -> output to layer 100 units
        self.relu2 = nn.ReLU()

        # hidden layer 3
        self.l4 = nn.Linear(100, 20)  # input shape 100 -> output to layer 20 units
        self.relu4 = nn.ReLU()

        # output layer
        self.l5 = nn.Linear(
            20, 3
        )  # input shape 20 -> output layer 3 units (wine/lose/draw classifier)

    def forward(self, X):
        out = self.l1(X)
        out = self.relu1(out)

        out = self.l2(out)
        out = self.relu2(out)

        out = self.l4(out)
        out = self.relu4(out)

        out = F.log_softmax(self.l5(out), dim=1)
        return out


dataset = ChessDataset()


def train():
    # prepare dataloader
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # train model
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    error = nn.CrossEntropyLoss()

    print("Training model...")
    correct, total = 0, 0
    for epoch in range(EPOCHS):
        for X, y in tqdm(train_dataloader):
            # clear gradients
            optimizer.zero_grad()

            # forward pass
            out = model(X.float())
            pred = torch.max(out, 1)[1]
            correct += torch.sum(pred == y).item()
            total += len(pred)

            # compute loss
            loss = error(out, y)

            # backprop
            loss.backward()

            # update parameters
            optimizer.step()

        # print train loss
        print(f"Epoch {epoch} Loss: {loss}")
        print(f"Train accuracy: {correct/total}")

    # store model on disk
    with open("processed/model.pickle", "wb") as fp:
        pickle.dump(model, fp)

    # evaluate model
    # print('Evaluating accuracy...')
    # correct, total = 0, 0
    # for X, y in tqdm(test_dataloader):
    #     y_preds = model(X.float())
    #     pred = torch.max(y_preds,1)[1]
    #     correct += torch.sum(pred==y).item()
    #     total += len(pred)
    # print(f'Test accuracy: {correct/total}')


if __name__ == "__main__":
    train()
