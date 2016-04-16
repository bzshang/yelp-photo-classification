"""
Custom metric for mxnet
"""

__author__ = 'bshang'

from sklearn.metrics import f1_score
from sklearn import preprocessing

def f1(label, pred):
    """ Custom evaluation metric on F1.
    """
    pred_bin = preprocessing.binarize(pred, threshold=0.5)
    score = f1_score(label, pred_bin, average='micro')
    return score
