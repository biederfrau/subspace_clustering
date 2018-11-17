import numpy as np
import pandas as pd


def read_file(file):
    data = []
    with open(file) as f:
        for _ in range(4):
            next(f)
        for line in f:
            current_line = line.strip().split(' ')
            current_line[0] = current_line[0][3:]
            data.append(current_line)

        data = np.array(data)

    return data

# Key: filename, Value: number of clusters
files = {"higher_dimensional": 4, "paper": 2, "test": 3, "test_noisy": 3}

for name in files:
    for iteration in range(1, 6):
        d = {}

        for i in range(files[name]):
            cluster = read_file(f"./elki_results/elki_{name}{iteration}/cluster_{i}.txt")
            d[i] = np.append(cluster, np.full((cluster.shape[0], 1), i), axis=1)  # append cluster label

        # Merge cluster data
        merged_data = d[0]
        for i in range(1, files[name]):
            merged_data = np.concatenate([merged_data, d[i]])

        # Sort data by ID (first column) & remove ID
        merged_data = np.array(sorted(merged_data, key=lambda x: int(x[0])))[:, 1:]

        # Write to csv file
        df = pd.DataFrame(merged_data)
        df.to_csv(f"./elki_csv_results/{name}{iteration}.csv", header=False, index=False)
