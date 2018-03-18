# ReID_PCB

Implementation of PCB model of paper: *Beyond Part Models: Person Retrieval with Refined Part Pooling.*

Using Python3.6 and Pytorch.

## Start

### Dataset Preparation

Support [Market1501 Dataset][http://www.liangzheng.org/Project/project_reid.html]. Change the dataset path at `prepare.py` and use

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

Using the default settings (remember to add `--train_all`, I train and test on Market1501 dataset and the results are:

**Top1: 90.77, Top5: 96.35, Top10: 97.56, mAP: 75.45**

Maybe there are some differences between my implementation and the paper's, which are not detailed.

