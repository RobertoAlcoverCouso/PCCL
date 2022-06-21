from pathlib import Path

from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms

class BaseDataset(data.Dataset):
    def __init__(self, root, list_path, set_,
                 max_iters, image_size, labels_size, mean, test=True, soft_label=False):
        self.root = Path(root)
        self.set = set_
        self.list_path = list_path.format(self.set)
        self.image_size = image_size
        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size
        self.mean = mean
        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
        self.soft_label = soft_label
        self.img_ids = self.img_ids
        self.files = []
        for name in self.img_ids:
            img_file, label_file = self.get_metadata(name)
            self.files.append((img_file, label_file, name))
        self.test = test
        self.transform = transforms.Compose([
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
                                ])
    def get_metadata(self, name):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def preprocess(self, image):
        #image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))

    def get_image(self, file):
        return _load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_labels(self, file):
        if self.soft_label:
            img = Image.open(file)
            return np.asarray(img, np.float32)
        return np.asarray(_load_img(file, self.labels_size, Image.NEAREST, rgb=False), np.float32)


def _load_img(file, size, interpolation, rgb):
    img = Image.open(file)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    return img

