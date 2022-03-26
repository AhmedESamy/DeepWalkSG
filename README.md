# DeepWalkSG

This repository contains the code for _Deepwalk: Online learning of social representations_, a paper in **Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014**.

## Contents

This repository includes:
* Edge lists and node labels of two heterogeneous graphs:
  - ACM
  - DBLP
* Code to generate random walks on those datasets using the following techniques:
  - DeepWalk
* Code to train SkipGram embeddings based on those random walks
* Code to evaluate the micro and macro F1 scores of those embeddings on the node classification task defined by the labels in the datasets

## Installation

To ensure reproducibility, the code was tested in python 3.8.10 using Pytorch with the following packages:
```
torch                   1.5.0               
torchvision             0.6.0 
```

## Usage

* To run the code, use the following script:
```
python main.py --dataset DBLP --number_walks 80 --window_size 5 --negative_samples 5 --walk_length 40 --dimension 128 --iterations 5 --workers 10 --batch_size 10000 --train 1
```

## How the Evaluation is Performed

Each embeddings file passed to `src/evaluate.py` is evaluated multiple times, on different data splits. Each data split is obtained by shuffling the data with one of the seeds in the chosen seed set (all seed sets are listed in `src/datasets.py`). The micro and macro F1 scores reported are the average across all data splits.

For each data split, the last 20% of the data is used for evaluating 8 classifiers, trained on the first 10%, 20%, ..., 80% of the data respectively. The micro and macro F1 scores are reported separately for each training data percentage.

## reference
_Perozzi, Bryan, Rami Al-Rfou, and Steven Skiena. "Deepwalk: Online learning of social representations." Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014_.
