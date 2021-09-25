import codecs
import pickle
import time, random
import numpy as np

import sys

sys.path.append("/Users/puriaradmard/Documents/GitHub/vectorai.2021")

from api.utils import producer_types
from api.interface import *


class ReducableDataSource:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.indices = list(range(len(data)))

    def __getitem__(self, idx):
        return {"X": self.data[idx], "y": self.labels[idx]}

    def select_data(self, idx):
        indices = set(self.indices)
        indices.remove(idx)
        self.indices = list(indices)
        return self[idx]


def generate_random_out_data(image_source: ReducableDataSource):
    chosen_index = random.choice(image_source.indices)
    chosen_data = image_source.select_data(chosen_index)
    chosen_image = chosen_data["X"]
    print(f"Sending image {chosen_index} of shape {chosen_image.shape}")
    outdata = (chosen_index, chosen_image.tolist())
    # pickled = pickle.dumps(outdata)
    # return codecs.encode(pickled, "base64").decode()
    return outdata


def image_stream(
    num_images, image_source: ReducableDataSource, topic: str, producer: producer_types
):
    waittimes = (abs(1 * np.random.randn(num_images)) + 2).astype(int)

    for wtime in waittimes:

        for t in list(range(wtime))[::-1]:
            print(f"Next image in {t}", end="\r")
            time.sleep(1)

        print(f"{len(image_source.indices)} images remaining")
        outdata = generate_random_out_data(image_source)
        producer.send_message(outdata, topic)
        print(f"{len(image_source.indices)} images remaining")


def correct_label(image_source: ReducableDataSource, chosen_index: int):
    return image_source.labels[chosen_index]


if __name__ == "__main__":
    random_data = np.random.randn(60000, 1, 28, 28)
    random_labels = np.random.randint(0, 10, [60000])
    random_data_source = ReducableDataSource(random_data, random_labels)

    config = Config(provider="kafka", server="localhost:9092")
    producer = Producer(config)
    producer.add_topic("fashion_mnist")

    image_stream(
        num_images=700,
        image_source=random_data_source,
        topic="fashion_mnist",
        producer=producer,
    )
