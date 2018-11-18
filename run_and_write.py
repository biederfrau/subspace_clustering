#!/usr/bin/env python3

import orclus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import argv, exit
from pathlib import Path
import random

import logging

from sklearn.metrics import normalized_mutual_info_score

random.seed(666)

def get(opt):# {{{
    idx = next((i for i, x in enumerate(argv) if x == opt), None)
    return argv[idx+1] if idx and idx+1 < len(argv) else None# }}}

f = argv[len(argv) - 1]
k = int(get('-k') or 2)
l = int(get('-l') or 2)

csv = pd.read_csv(f, delim_whitespace=True, comment='#', header=None)
mat = np.matrix(csv.values[:, 0:csv.shape[1]-1], dtype=np.float32)

nmis = []
predictions = []
for i in range(5):
    clusters, seeds, vectors = orclus.orclus(mat, k, l, k0=80 if Path(f).stem == 'higher_dimensional' else 50)
    pred = orclus.predict(mat, seeds, vectors)

    predictions.append(pred)

    nmi = normalized_mutual_info_score(csv[csv.shape[1] - 1], pred)
    nmis.append(nmi)

    print(f"{'='*32}\nNMI score: {nmi}")

print(f"{'='*32}\nAverage NMI score: {np.average(nmis)}")

with open(f"{Path(f).stem}_NMI.txt", "w+") as fh:
    for nmi in nmis:
        fh.write(f"{nmi}\n")

with open(f"{Path(f).stem}_best_clustering.txt", "w+") as fh:
    for c in predictions[np.argmax(nmi)]:
        fh.write(f"{c}\n")
