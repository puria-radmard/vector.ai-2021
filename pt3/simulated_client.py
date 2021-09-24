import time, torch, random
from cnn.utils import SimpleImageDataset


class ReducableDataSource:
    def __init__(self, data, labels):
        self.dataset = SimpleImageDataset(data, labels)
        self.indices = list(range(len(data)))

    def select_data(self, idx):
        indices = set(self.indices)
        indices.remove(idx)
        self.indices = list(indices)
        return self.dataset[idx]



def image_stream(num_images, image_source: ReducableDataSource):

    waittimes = (abs(5*torch.randn(num_images)).int() + 5).cpu().numpy()

    for wtime in waittimes:

        for t in list(range(wtime))[::-1]:

            print(f"Next image in {t}", end='\r')
            time.sleep(1)
            
        chosen_index = random.choice(image_source.indices)
        chosen_data = image_source.select_data(chosen_index)
        chosen_image, chosen_label = chosen_data["X"], chosen_data["y"]
        print(f"Sending image {chosen_index} of shape {chosen_image.shape}, class {chosen_label.item()}")
        print(f"{len(image_source.indices)} images remaining")


if __name__ == '__main__':
    random_data = torch.randn(60000, 1, 28, 28)
    random_labels = torch.randint(0, 10, [60000])
    random_data_source = ReducableDataSource(random_data, random_labels)
    image_stream(num_images = 700, image_source = random_data_source)
