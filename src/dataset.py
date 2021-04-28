import torch
from PIL import Image
import pandas as pd
import os
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset

train_image_locations = 'src/data/train_image_locations.txt'
validation_image_locations = 'src/data/validation_image_locations.txt'
test_image_locations = 'src/data/test_image_locations.txt'
train_annotations = 'src/data/train_annotations.csv'
validation_annotations = 'src/data/validation_annotations.csv'
test_annotations = 'src/data/test_annotations.csv'


def collate_fn(batch):
    """Makes a batch"""
    items = list(zip(*batch))
    # img, imgid, bbox, label
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    items[1] = list([i for i in items[1] if i])
    items[2] = default_collate([i for i in items[2] if torch.is_tensor(i)])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    return items


class OIDataset(Dataset):
    def __init__(self, transform=None, train=False, validation=False, test=False):
        self.transform = transform
        self.train = train
        self.validation = validation
        self.test = test

    def _get_img_name(self, path):
        """Converts path to image to image id"""
        return os.path.splitext(os.path.split(path)[1])[0]

    def __len__(self):
        df = None
        if self.train:
            df = pd.read_csv(train_annotations)
        elif self.validation:
            df = pd.read_csv(validation_annotations)
        elif self.test:
            df = pd.read_csv(test_annotations)
        return len(df)

    def __getitem__(self, index):
        """Gets image and annotation at a given index"""
        image = None

        path_to_images, path_to_annotations = None, None
        if self.train:
            path_to_images = train_image_locations
            path_to_annotations = train_annotations
        elif self.validation:
            path_to_images = validation_image_locations
            path_to_annotations = validation_annotations
        elif self.test:
            path_to_images = test_image_locations
            path_to_annotations = test_annotations

        with open(path_to_images) as file:
            for i, line in enumerate(file):
                if i == index:
                    path_to_image = str.rstrip(line)
                    image = Image.open(path_to_image).convert("RGB")

        if image is None:
            return None, None, None, None

        df = pd.read_csv(path_to_annotations)
        line = df.loc[df['imgid'] == self._get_img_name(path_to_image)].to_numpy()
        label_id = torch.tensor([line[0][5]])  # 1 2 3 instead of 0 1 2 (0 is background)
        # xmin, ymin, xmax, ymax
        bbox = torch.tensor([[line[0][1], line[0][2], line[0][3], line[0][4]]])
        if self.transform is not None:
            image, bbox, label_id = self.transform(image, bbox, label_id)
        # opened image, id, bbox, label id
        return image, line[0][0], bbox, label_id
