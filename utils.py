#!/usr/bin/env python3

import numpy as np

import scipy
import sklearn

from scipy.io import arff
from sklearn import ensemble, model_selection, metrics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def arff_to_ndarray(f, ycol=None):
    data, meta = scipy.io.arff.loadarff(f)
    nom_names = [name for name, typ in zip(meta.names(), meta.types()) if typ == 'nominal']
    num_names = [name for name, typ in zip(meta.names(), meta.types()) if typ == 'numeric']

    if len(nom_names) > 1 and ycol is None:
        raise ValueError("more than one nominal column---specify column")

    if not len(nom_names):
        raise ValueError("no nominal data. wat")

    if not len(num_names):
        raise ValueError("no numerical data. wat")

    if ycol is not None:
        Y = data[ycol]
    else:
        Y = data[nom_names[0]]

    D = data[num_names] \
            .copy() \
            .view(None) \
            .reshape(-1, len(num_names))

    Y = Y.reshape(-1, 1)

    return (D, Y)
