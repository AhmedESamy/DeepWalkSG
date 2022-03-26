import pandas as pd
import numpy as np

from utils.graph import nodes_to_idx
from six import iteritems
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle as skshuffle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split



class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


def _logistic_regression(x_train, x_test, y_train, y_test):
    clf = TopKRanker(LogisticRegression(random_state=42, max_iter=1000))
    clf.fit(x_train, y_train)

    num_classes = y_test.shape[1]
    top_k_list = [len([v for v in yi if v]) for yi in y_test] # find out how many labels should be predicted
    predications = clf.predict(x_test, top_k_list)

    return transform_to_binary(predications, num_classes)


def transform_to_binary(data, num_classes):
    mlb = MultiLabelBinarizer(np.arange(num_classes))
    return mlb.fit_transform(data)


def get_embeddings(keys, embeddings):
    return np.float_([embeddings[key] for key in keys])


def predict(x, x_test, y, y_test, random_seeds, num_shuffles= 10):

    shuffles = []
    all_results = defaultdict(list) # to score each train group
    data_size = x.shape[0] + x_test.shape[0]
    training_percents = np.asarray(range(1, 9)) * .1  # training percentages

    for i in range(num_shuffles): # Shuffle: should be producible in future comparisons.
        shuffles.append(skshuffle(x, y, random_state=random_seeds[i]))

    for train_percent in training_percents:
        training_size = int(train_percent * data_size)  # training size 80% of the original data

        for shuffle in shuffles:
            x_data, y_data = shuffle
            x_train = x_data[:training_size, :]
            y_train = y_data[:training_size]

            results = {}
            predictions = _logistic_regression(x_train, x_test, y_train, y_test)

            averages = ["micro", "macro"]
            for average in averages:
                results[average] = f1_score(y_test, predictions, average=average)
            all_results[train_percent].append(results)

    for train_percent in sorted(all_results.keys()):
        avg_score = defaultdict(float)
        for score_dict in all_results[train_percent]:
            for metric, score in iteritems(score_dict):
                avg_score[metric] += score
        for metric in avg_score:
            avg_score[metric] /= len(all_results[train_percent])

        print(list(dict(avg_score).values()))


def k_fold_cross_validation(x, y, k=5):
    all_results = list()  # to score each train group
    kf = KFold(n_splits=k)  # k fold cross validation

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        results = {}
        predictions = _logistic_regression(x_train, x_test, y_train, y_test)

        averages = ["micro", "macro"]
        for average in averages:
            results[average] = f1_score(y_test, predictions, average=average)
        all_results.append(results)

    avg_score = defaultdict(float)
    for score_dict in all_results:
        for metric, score in iteritems(score_dict):
            avg_score[metric] += score
    for metric in avg_score:
        avg_score[metric] /= kf.n_splits

    print(list(dict(avg_score).values()))


def node_classification(graph, embeddings, path_to_labels, random_seeds, sep='\t', eval=False):
    reader = pd.read_csv(path_to_labels, header=None, sep=sep)

    y = reader.iloc[:, 1:].to_numpy()
    xi = nodes_to_idx(graph, np.array(list(reader.iloc[:, 0])))
    x = get_embeddings(xi, embeddings) if type(embeddings) is dict else embeddings[xi]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_seeds[0][0])

    if eval:
        predict(x_train, x_test, y_train, y_test, random_seeds[0])
        return

    k_fold_cross_validation(x_train, y_train)

