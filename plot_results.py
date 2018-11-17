#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv
from pathlib import Path

colors = ['r', 'g', 'gold', 'b']
col = { "cluster1": "r", "cluster2": "g", "cluster3": "gold", "cluster4": "b", "noise": "grey" }

f = argv[1]
thing = Path(f).stem[:-1]
data_elki = pd.read_csv(f, header=None)

cmap = list(map(lambda x: colors[x], data_elki[data_elki.shape[1] - 1]))

fig = plt.figure(figsize=(16, 6) if thing == "test" or thing == "test_noisy" else (11, 6))
ax = fig.add_subplot(131 if thing == "test" or thing == "test_noisy" else 121)

ax.scatter(data_elki[0], data_elki[1], c=cmap)
ax.set_title("ELKI")
ax.set_xlabel("dimension 1")
ax.set_ylabel("dimension 2")

labels = pd.read_csv(f"results/{thing}_best_clustering.txt", header=None, sep=" ")
cmap = list(map(lambda x: colors[x], labels[0]))

ax = fig.add_subplot(132 if thing == "test" or thing == "test_noisy" else 122)
ax.scatter(data_elki[0], data_elki[1], c=cmap)
ax.set_title("DM")
ax.set_xlabel("dimension 1")
ax.set_ylabel("dimension 2")

if thing == "test" or thing == "test_noisy":
    data = pd.read_csv(f"data/{thing}.csv", delim_whitespace=True, comment='#', header=None)
    cmap = list(map(lambda x: col[x], data[data.shape[1] - 1]))

    ax = fig.add_subplot(133)
    ax.scatter(data[0], data[1], c=cmap)

    ax.set_title("Ground Truth")
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")

plt.tight_layout()

plt.show()
