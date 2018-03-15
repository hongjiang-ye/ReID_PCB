# ReID_baseline

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

### Test

Set the hyperparameters in the same way.

```python
python test.py
```

The code loads the trained model and extracts the features of testing data. CMC(Top 1, Top 5, Top 10) and mAP are provided for evaluation.

## Result

Using the default settings (remember to add `--train_all`, we got:

**Top1: 89.81, Top5: 96.08, Top10: 97.44, mAP: 74.66**

