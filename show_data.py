#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import pandas as pd

col = { "cluster1": "b", "cluster2": "r", "cluster3": "g", "cluster4": "y" }

f = sys.argv[1]
data = pd.read_csv(f, delim_whitespace=True, comment='#', header=None)

plt.scatter(data[0], data[1], c=list(map(lambda x: col[x], data[data.shape[1] - 1])))
plt.show()
