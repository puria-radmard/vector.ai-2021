import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F


class ReluLayer(nn.Module):
    def __init__(self):
        super(self, ReluLayer).__init__()

    def forward(self, x):
        return F.relu(x)


class FlattenLayer(nn.Module):
    def __init__(self):
        super(self, FlattenLayer).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)


def generate_convolutional_layers(channel_sequence, kernel_sizes):
    convolutional_layers = []
    [
        convolutional_layers.extend(
            [
                nn.Conv2d(
                    channel_sequence[i], channel_sequence[i + 1], kernel_sizes[i]
                ),
                ReluLayer(),
                nn.MaxPool2d(2, 2),
            ]
            for i in range(len(channel_sequence) - 1)
        )
    ]
    convolutional_layers = nn.ModuleList(convolutional_layers)
    return convolutional_layers


def generate_fc_layers(conv_output_size, hidden_fc_sequence, num_classes):
    fc_layers = [nn.Linear(conv_output_size, hidden_fc_sequence[0])]
    [
        fc_layers.extend(
            [nn.Linear(hidden_fc_sequence[i], hidden_fc_sequence[i + 1]), ReluLayer()]
            for i in range(len(hidden_fc_sequence) - 1)
        )
    ]
    fc_layers.append(nn.Linear(hidden_fc_sequence[-1], num_classes))
    fc_layers = nn.ModuleList(fc_layers)
    return fc_layers


def top_1_accuracy(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    correct = sum(preds == labels).item()


def epoch_process(model, criterion, optimizer, dataloader, mode):

    assert mode in ["train", "eval"]

    epoch_loss = 0
    total_correct = 0
    counter = 0

    for i, data in tqdm(dataloader):
        if mode == "train":
            optimizer.zero_grad()

        inputs, labels = data["X"], data["y"]
        print("check inputs size 0 is actually batch_size")
        import pdb

        pdb.set_trace()
        batch_size = inputs.size[0]

        output = model(inputs)
        loss = criterion(output, labels)
        if mode == "train":
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        total_correct += top_1_accuracy(output, labels)
        counter += batch_size

    return epoch_loss, total_correct, counter
