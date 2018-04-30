# ReID_PCB

Implementation of PCB model of paper: *Beyond Part Models: Person Retrieval with Refined Part Pooling.*

Using Python3.6 and **Pytorch 4.0**.

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

to transform the dataset to Pytorch ImageFolder API style. 

### Train

Set the parameters by command line `--epochs 30` and specify the training dataset, or change the default values in the code.

```python
python train.py --dataset {market1501, duke, cuhk03}
```

![train_log](https://ws3.sinaimg.cn/large/006tKfTcgy1fquuibr5z4j30hs0dcwfc.jpg)

### Test

Set the hyperparameters in the same way.

```python
python test.py --dataset {market1501, duke, cuhk03}
```

The code loads the trained model and extracts the features of testing data. CMC(Top 1, Top 5, Top 10) and mAP are provided for evaluation.

## Result

Using the default settings, the results are:

|                                      | Top1   | Top 5  | Top 10 | mAP    |
| ------------------------------------ | ------ | ------ | ------ | ------ |
| Market1501 (share 1x1 conv)          | 91.51  | 96.91  | 98.01  | 75.81  |
| Market1501 (independent 1x1 conv)    | 93.08  | 97.15  | 97.92  | 79.57  |
| *Market1501(Paper)*                  | *92.4* | *97.0* | *97.9* | *77.3* |
| DukeMTMC-reID (share 1x1 conv)       | 81.55  | 90.40  | 93.00  | 67.03  |
| DukeMTMC-reID (independent 1x1 conv) | 84.25  | 91.83  | 94.03  | 71.06  |
| *DukeMTMC-reID(Paper)*               | *81.9* | *89.4* | *91.6* | *65.3* |
| CUHK03 (share 1x1 conv)              | 46.29  | 67.14  | 76.14  | 44.41  |
| CUHK03 (independent 1x1 conv)        | 56.57  | 74.79  | 82.14  | 53.88  |
| *CUHK03(paper)*                      | *61.3* | *78.6* | *85.6* | *54.2* |

Testing results on CUHK03 are much lower than those reported on the paper. Maybe there are some differences between my implementation and the paper's, which are not detailed. Please inform me whatever you found useful.

