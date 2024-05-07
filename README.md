#Code for PCCL [[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4410425)].
Based on the [[ADVENT](https://github.com/valeoai/ADVENT)] paper and code.

## Preparation

### Pre-requisites
* Python 3.7
* Pytorch >= 0.4.1
* CUDA 9.0 or higher

#### Data downlading
By default, the datasets are put in ```<root_dir>/../data```. We use symlinks to hook the ADVENT codebase to the datasets. An alternative option is to explicitlly specify the parameters ```DATA_DIRECTORY_SOURCE``` and ```DATA_DIRECTORY_TARGET``` in YML configuration files.

* **GTA5**: Please follow the instructions [here](https://download.visinf.tu-darmstadt.de/data/from_games/) to download images and semantic segmentation annotations. The GTA5 dataset directory should have this basic structure:
```bash
<root_dir>/data/GTAV/                               % GTA dataset root
<root_dir>/data/GTAV/images/                        % GTA images
<root_dir>/data/GTAV/labels/                        % Semantic segmentation labels
...
```

* **Cityscapes**: Please follow the instructions in [Cityscape](https://www.cityscapes-dataset.com/) to download the images and validation ground-truths. The Cityscapes dataset directory should have this basic structure:
```bash
<root_dir>/data/Cityscapes/                         % Cityscapes dataset root
<root_dir>/data/Cityscapes/leftImg8bit              % Cityscapes images
<root_dir>/data/Cityscapes/leftImg8bit/val
<root_dir>/data/Cityscapes/gtFine                   % Semantic segmentation labels
<root_dir>/data/Cityscapes/gtFine/val
...
```
### Installation
0. Clone the repo:
```bash
$ git clone https://github.com/RobertoAlcoverCouso/PCCL/
$ cd ADVENT
```

1. Install OpenCV if you don't already have it:

```bash
$ conda install -c menpo opencv
```

2. Install this repository and the dependencies using pip:
```bash
$ pip install -e <root_dir>
```

With this, you can edit the code on the fly and import function 
and classes of ADVENT in other project as well.

3. Optional. To uninstall this package, run:
```bash
$ pip uninstall ADVENT
```
### Check Installation
You'll need at least image "frankfurt/frankfurt_000001_007973_leftImg8bit.png" in folder <root_dir>/data/Cityscapes/leftImg8bit and a download test_model.pth  in  <root_dir>/advent
```bash
$ cd <root_dir>/advent/scripts
$ python test.py --cfg configs/check.yml
```
### Training
For the experiments done in the paper, we used pytorch 0.4.1 and CUDA 9.0. To ensure reproduction, the random seed has been fixed in the code. However, you can select a random seed indicating with the flag --random-train

By default, logs and snapshots are stored in ```<root_dir>/experiments``` with this structure:
```bash
<root_dir>/experiments/logs
<root_dir>/experiments/snapshots
```
#### Training source model
```bash
$ cd <root_dir>/advent/scripts
$ python train.py --cfg ./configs/Pre_defined.yml
$ python train.py --cfg ./configs/Pre_defined.yml --tensorboard         % using tensorboard
```
#### Training UDA model
For a given UDA method ``method":
```bash
$ cd <root_dir>/advent/scripts
$ python train.py --cfg ./configs/method.yml
$ python train.py --cfg ./configs/method.yml --tensorboard         % using tensorboard
```
If multiple GPUs are available, you can denote it by:
```bash
$ python train.py --cfg ./configs/method.yml -n N -g G -r R
```
where N is the number of nodes, G the number of gpus per node and r the rank of each process.
make sure to tune the variables: os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] to your machine.

### Testing
To test a given method ``method":
```bash
$ cd <root_dir>/advent/scripts
$ python test.py --cfg ./configs/method.yml
```
## Important files
advent/domain_adaptation/train_UDA.py contains the implementation of each method.

advent/model/* contains the implementation of DeplabV2 and the discriminator.

advent/scripts/train.py contains the higher level code for defining the datasets, loaders and training method.
