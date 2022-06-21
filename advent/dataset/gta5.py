import numpy as np
from advent.domain_adaptation.config import cfg
from advent.dataset.base_dataset import BaseDataset
import torchvision.transforms.functional as TF
from torch.utils import data
from tqdm import tqdm
import torch
import csv
import pandas as pd
from torchvision.utils import make_grid

from advent.utils.viz_segmask import color_mapping, show


class GTA5DataSet(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), test=False, files_to_use=None):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean, test)

        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.len = len(self)
        if files_to_use:
            self.files_to_use = pd.read_csv(files_to_use)
            self.proportions = self.files_to_use.iloc[:,1:].sum()/316748
        self.class_to_include = 255
        

    def get_metadata(self, name):
        img_file = self.root / 'images' / name
        label_file = self.root / 'labels' / name
        return img_file, label_file

    def include_class(self, clases):
        self.class_to_include = clases
        self.files = []
        if isinstance(clases, list):
            clas = self.files_to_use[clases].any(1)
        else:
            clas = self.files_to_use[clases]   
        for name in self.files_to_use["name"][clas]:
            img_file, label_file = self.get_metadata(name)
            self.files.append((img_file, label_file, name))
        self.len = len(self.files)
    def get_n_iters(self, clases, total):
        if isinstance(clases, list):
            clas = self.files_to_use[clases].any(1)
        else:
            clas = self.files_to_use[clases]
        return int((self.proportions[clases]*total).sum().round())
    def __getitem__(self, index):
        
        img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        image = self.transform(image)
        image = np.asarray(image, np.float32)
        label = self.get_labels(label_file)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        image = self.preprocess(image)
        

        if np.random.rand() > 0.5:
            image = image[:,:, ::-1]
            label_copy = label_copy[:, ::-1]
        return image.copy(), label_copy.copy(), np.array(image.shape), name
def main():

    source_dataset = GTA5DataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                 list_path=cfg.DATA_LIST_SOURCE,
                                 set=cfg.TRAIN.SET_SOURCE,
                                 crop_size=(320, 180),
                                 mean=cfg.TRAIN.IMG_MEAN,
                                 test=False)
    source_loader = data.DataLoader(source_dataset,
                                        batch_size=1,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=False,
                                        pin_memory=True,
                                        worker_init_fn=None)
    #pool = torch.nn.AvgPool2d(3)
    #pixels_dict ={i:0 for i in range(19)}
    #labels_included = {"name":[]}
    #labels_included.update({lab:[] for lab in cs_labels}) 
    
    for batch in tqdm(source_loader):
        images_source, labels, _, name = batch
        
        grid_image = make_grid(image[0].clone().cpu().data, 3, normalize=True)
        grid_gt = make_grid(torch.from_numpy(color_mapping(labels).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
        grid = make_grid([grid_gt, grid_prediction])
        show(grid, name)
        continue
        exit(1)
        labels_included["name"].append(name)  
        for i in range(19):
            labels_included[cs_labels[i]].append(i in labels) 
        continue
        labels[labels == 255] = 19
        labels[labels < 0] = 19 
        
        oh = torch.nn.functional.one_hot(labels.long(), 20). float().cuda()
        # print(oh.shape)
        res = pool(oh.permute(0,3,1,2))
        #print(res.shape)
        for i in range(19):
            pixels_dict[i] += (res[:,i, :,:] != 1).sum()/(res[:,i, :,:].sum())
            #labels = labels.long().to(0)
            #pixels_dict[i] += (labels==i).sum()
    
    #df = pd.DataFrame(labels_included)
    #df.to_csv('gta5.csv', index=False)
    #print(pixels_dict)
if __name__ == '__main__':
    main()