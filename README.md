# ReID_PCB

Implementation of PCB model of paper: *Beyond Part Models: Person Retrieval with Refined Part Pooling.*

Using Python3.6 and Pytorch.

## Start

### Dataset Preparation

Support Market1501, DukeMTMC-reID and CUHK03 dataset. 

Download Market1501 dataset from [here](http://www.liangzheng.org/Project/project_reid.html).

Download DukeMTMC-reID dataset from [here](https://github.com/layumi/DukeMTMC-reID_evaluation).

Download CUHK03 dataset from [here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html). You should also download "cuhk03_new_protocol_config_detected.mat" from [here](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03) and put it with `cuhk-03.mat`. We meed this new protocol to split the dataset.

Please unzip the file and not change any of the directory names. Then change the paths in `utils.py` to the unzipped directory path, such as `/Users/ruby/Downloads/Market-1501-v15.09.15` and run

```python
python data_transform.py --dataset {market1501, duke, cuhk03}
```

to transform the dataset to Pytorch API style. 

### Train

Set the parameters by command line `--epochs 30` and specify the training dataset, or change the default values in the code.

```python
python train.py --dataset {market1501, duke, cuhk03}
```

The code will save the trained model at `./model/{dataset}`.

![train_log](https://i.loli.net/2018/03/19/5aaf1664bae2b.jpg)

Using the hyperparameters from the paper, the loss converage at around 30th epoch. 

### Test

Set the hyperparameters in the same way.

```python
python test.py --dataset {market1501, duke, cuhk03}
```

The code loads the trained model and extracts the features of testing data. CMC(Top 1, Top 5, Top 10) and mAP are provided for evaluation.

## Result

Using the default settings (remember to add `--train_all` !), the results are:

|                        | Top1   | Top 5  | Top 10 | mAP    |
| ---------------------- | ------ | ------ | ------ | ------ |
| Market1501             | 91.51  | 96.91  | 98.01  | 75.81  |
| *Market1501(Paper)*    | *92.4* | *97.0* | *97.9* | *77.3* |
| DukeMTMC-reID          | 81.55  | 90.40  | 93.00  | 67.03  |
| *DukeMTMC-reID(Paper)* | *81.9* | *89.4* | *91.6* | *65.3* |
| CUHK03                 | 46.29  | 67.14  | 76.14  | 44.41  |
| *CUHK03(paper)*        | *61.3* | *78.6* | *85.6* | *54.2* |

Testing results on CUHK03 are much lower than those reported on the paper. Maybe there are some differences between my implementation and the paper's, which are not detailed. Please inform me whatever you found useful.

