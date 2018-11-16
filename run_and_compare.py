#!/usr/bin/env python3

import orclus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.metrics import normalized_mutual_info_score

f = sys.argv[1]
csv = pd.read_csv(f, delim_whitespace=True, comment='#', header=None)

colors = ["b", "r", "g", "y"]

mat = np.matrix(csv.values[:, 0:2], dtype=np.float32)

clusters, seeds, vectors = orclus.orclus(mat, 2, 2, k0=50)
pred = orclus.predict(mat, seeds, vectors)

nmi = normalized_mutual_info_score(csv[csv.shape[1] - 1], pred)
print(f"{'='*32}\nNMI score: {nmi}")

plt.scatter(np.asarray(mat[:, 0]), np.asarray(mat[:, 1]), c=list(map(lambda x: colors[x], pred)))
plt.show()
