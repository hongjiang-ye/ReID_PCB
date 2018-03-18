# ReID_PCB

Implementation of PCB model of paper: *Beyond Part Models: Person Retrieval with Refined Part Pooling.*

Using Python3.6 and Pytorch.

## Start

### Dataset Preparation

Support Market1501 and DukeMTMC-reID Dataset. Change the dataset path at `prepare.py` such as '/Users/ruby/Downloads/Market-1501-v15.09.15' and use

```python
python prepare.py
```

to convert the dataset to Pytorch style. 

### Train

Set the arguments and hyperparameters by command line `--epochs 30` or change the default values in the code.

```python
python train.py
```

The code will save the trained model at `./model`.

![train_log](https://i.loli.net/2018/03/18/5aadfe782bed6.jpg)

Using the hyperparameters from the paper, the loss converage at around 30th epoch. 

### Test

Set the hyperparameters in the same way.

```python
python test.py
```

The code loads the trained model and extracts the features of testing data. CMC(Top 1, Top 5, Top 10) and mAP are provided for evaluation.

## Result

Using the default settings (remember to add `--train_all`), the results are:

* Market1501: **Top1: 90.77, Top5: 96.35, Top10: 97.56, mAP: 75.45**

  (paper: Top1: 92.4, Top5: 97.0, Top10: 97.9, mAP: 77.3)

DukeMTMC-reID: **Top1: 80.75, Top5: 89.41, Top10: 92.10, mAP: 66.63**

（paper: top1: 81.9, top5: 89.4, top10: 91.6, mAP: 65.3）

Maybe there are some differences between my implementation and the paper's, which are not detailed.

