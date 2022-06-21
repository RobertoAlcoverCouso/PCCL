import numpy as np
from advent.domain_adaptation.config import cfg
from torch.utils import data
from advent.utils import project_root
from advent.utils.serialization import json_load
from advent.dataset.base_dataset import BaseDataset
import torchvision.transforms.functional as TF
from torchvision.transforms import functional as F
from tqdm import tqdm
import shutil
import csv
import pandas as pd

DEFAULT_INFO_PATH = project_root / 'advent/dataset/cityscapes_list/info.json'
cs_labels = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "light",
    "sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motocycle",
    "bicycle"]

class CityscapesDataSet(BaseDataset):
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path=DEFAULT_INFO_PATH, labels_size=None, test=True):
        self.pseudolabels = pseudolabels
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean, test)

        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str)
        self.mapping = np.array(self.info['label2train'], dtype=np.int)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label
        

    def get_metadata(self, name):
        img_file = self.root / 'leftImg8bit' / self.set / name
        label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        label_file = self.root / 'gtFine' / self.set / label_name
        
        return img_file, label_file


    def map_labels(self, input_):
        if self.pseudolabels:
            return input_
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        label = self.get_labels(label_file)
        label = self.map_labels(label)
        image = self.get_image(img_file)
        if self.pseudolabels:
            image = self.transform(image)
        image = np.asarray(image, np.float32)
        image = self.preprocess(image)
        if not self.test:
            if np.random.rand() > 0.5:
                image = image[:,:, ::-1]
                label = label[:,::-1]
            
        return image.copy(), label.copy(), np.array(image.shape), name
