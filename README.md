#Code for PCCL [[Paper]([https://arxiv.org/abs/2303.11797](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4410425)].
Based on the [[ADVENT](https://github.com/valeoai/ADVENT)] paper and code.

## Preparation

### Pre-requisites
* Python 3.7
* Pytorch >= 0.4.1
* CUDA 9.0 or higher

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

With this, you can edit the ADVENT code on the fly and import function 
and classes of ADVENT in other project as well.

3. Optional. To uninstall this package, run:
```bash
$ pip uninstall ADVENT
```

### Training
For the experiments done in the paper, we used pytorch 0.4.1 and CUDA 9.0. To ensure reproduction, the random seed has been fixed in the code. Still, you may need to train a few times to reach the comparable performance.

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
### Testing
To test a given method ``method":
```bash
$ cd <root_dir>/advent/scripts
$ python test.py --cfg ./configs/method.yml