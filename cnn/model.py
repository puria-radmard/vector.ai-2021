from torch import nn
from torch.nn import functional as F

from cnn.utils import *
from cnn.config import *


__all__ = ["ConvNN", "generate_model_by_name"]


class ConvNN(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        channel_sequence,
        kernel_sizes,
        hidden_fc_sequence,
    ):

        super(ConvNN, self).__init__()

        assert len(channel_sequence) - 1 == len(
            kernel_sizes
        ), "Require kernel_sizes length 1 less than channel_sequence length"
        # ADD CONV TO FC CHECK HERE
        assert (
            len(input_size) == 3 and input_size[0] == channel_sequence[0]
        ), f"Input size must be 3 dimensional, and first dimension ({input_size[0]}) must be the same as channel_sequence[0] ({channel_sequence[0]})"

        self.input_size = input_size
        self.channel_sequence = channel_sequence
        self.kernel_sizes = kernel_sizes
        self.hidden_fc_sequence = hidden_fc_sequence
        self.num_classes = num_classes

        self.convolutional_layers = generate_convolutional_layers(
            channel_sequence, kernel_sizes
        )
        self.flatten = FlattenLayer()
        self.conv_output_size = compute_output_size(
            self.input_size, self.convolutional_layers, self.flatten
        )
        self.fc_layers = generate_fc_layers(
            self.conv_output_size, hidden_fc_sequence, num_classes
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


def generate_model_by_name(num_classes, input_size, name="default") -> nn.Module:
    cs = CHANNEL_SEQUENCE_DICT[name]
    ks = KENREL_SIZES_DICT[name]
    fc = FC_SEQUENCE_DICT[name]
    model = ConvNN(
        num_classes=num_classes,
        input_size=input_size,
        channel_sequence=cs,
        kernel_sizes=ks,
        hidden_fc_sequence=fc,
    )
    return model
