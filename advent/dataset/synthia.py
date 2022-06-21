import numpy as np

from advent.dataset.base_dataset import BaseDataset
import torchvision.transforms.functional as TF

class SynthiaDataSet(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)

        # map to cityscape's ids
        self.id_to_trainid = {1:10, 2:2, 3:0, 4:1, 5:4, 6:8, 7:5, 8:13, 
                            9:7, 10:11, 11:18, 12:17, 15:6, 16:9, 17:12, 
                            18:14, 19:15, 20:16, 21:3}

    def get_metadata(self, name):
        img_file = self.root / 'images' / name
        label_file = self.root / 'labels' / name
        return img_file, label_file

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        image = self.preprocess(image)
        if np.random() > 0.5:
            image = TF.hflip(image)
            label  = TF.hflip(label)
        return image.copy(), label_copy.copy(), np.array(image.shape), name