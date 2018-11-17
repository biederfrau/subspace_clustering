#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

col = { "cluster1": "r", "cluster2": "g", "cluster3": "gold", "cluster4": "b", "noise": "grey" }

f = sys.argv[1]
data = pd.read_csv(f, delim_whitespace=True, comment='#', header=None)

colors = list(map(lambda x: col[x], data[data.shape[1] - 1]))

if data.shape[1] == 5:
    fig = plt.figure(figsize=(11, 6))
    fig.suptitle(f)

    ax = fig.add_subplot(121)
    ax.scatter(data[0], data[1], c=colors)
    ax.set_xlabel('dimension 1')
    ax.set_ylabel('dimension 2')

    ax = fig.add_subplot(122)
    ax.scatter(data[2], data[3], c=colors)
    ax.set_xlabel('dimension 3')
    ax.set_ylabel('dimension 4')
else:
    fig = plt.figure(figsize=(7, 6))
    fig.suptitle(f)

    ax = fig.add_subplot(111)

    ax.scatter(data[0], data[1], c=colors)
    ax.set_xlabel('dimension 1')
    ax.set_ylabel('dimension 2')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"img/{Path(f).stem}.pdf")
plt.show()
