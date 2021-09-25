import torch
import numpy as np
from cnn.model import generate_model_by_name, ConvNN

__all__ = ["load_model", "predict_numpy_images"]


def load_model(num_classes, input_size, name, weight_path):
    model = generate_model_by_name(num_classes, input_size, name)
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)
    return model


def predict_numpy_images(numpy_images, model):
    # numpy_image_shapes = [numpy_image.shape for numpy_image in numpy_images]
    # assert all(
    #     len(numpy_image_shape) == 3 for numpy_image_shape in numpy_image_shapes
    # ), "Numpy image must all be 3 dimensional"

    # THIS MIGHT MESS UP
    # TODO: CHECK/FIX THIS
    d_type = list(model.parameters())[0].dtype
    reshaped_numpy_image = np.stack(numpy_images)
    image_batch = torch.tensor(reshaped_numpy_image, dtype=d_type)
    model.eval()
    results = model(image_batch)
    return results
