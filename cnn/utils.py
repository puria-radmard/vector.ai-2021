import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "ReluLayer",
    "FlattenLayer",
    "generate_convolutional_layers",
    "generate_fc_layers",
    "top_1_accuracy",
    "compute_output_size",
    "SimpleImageDataset",
    "coll_fn",
]


class ReluLayer(nn.Module):
    def __init__(self):
        super(ReluLayer, self).__init__()

    def forward(self, x):
        return F.relu(x)


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

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
        )
        for i in range(len(channel_sequence) - 1)
    ]
    convolutional_layers = nn.Sequential(*convolutional_layers)
    return convolutional_layers


def generate_fc_layers(conv_output_size, hidden_fc_sequence, num_classes):
    fc_layers = [nn.Linear(conv_output_size, hidden_fc_sequence[0])]
    [
        fc_layers.extend(
            [nn.Linear(hidden_fc_sequence[i], hidden_fc_sequence[i + 1]), ReluLayer()]
        )
        for i in range(len(hidden_fc_sequence) - 1)
    ]
    fc_layers.append(nn.Linear(hidden_fc_sequence[-1], num_classes))
    fc_layers = nn.Sequential(*fc_layers)
    return fc_layers


def compute_output_size(input_size, *modules):
    example_input = torch.randn(1, *input_size)
    example_output = nn.Sequential(*modules)(example_input)
    example_output_shape = example_output.shape
    assert (
        len(example_output_shape) == 2 and example_output_shape[0] == 1
    ), f"Error in convolutional + flatten layers: output of flatten should be of shape [1, _], but is of shape {example_output_shape}"
    return example_output_shape[-1]


def top_1_accuracy(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    correct = sum(preds == labels).item()
    return correct


class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        super(SimpleImageDataset, self).__init__()
        self.data = data
        self.labels = labels
        assert len(self.labels) == len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"X": self.data[idx], "y": self.labels[idx]}


def coll_fn(instances):
    return {
        "X": torch.stack([ins["X"] for ins in instances]),
        "y": torch.stack([ins["y"] for ins in instances]),
    }
