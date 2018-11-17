import numpy as np
from utils import arff_to_ndarray
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score


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


# Load cluster files
cluster0 = read_file("./elki_output/cluster_0.txt")
cluster1 = read_file("./elki_output/cluster_1.txt")

# Create cluster labels
label0 = np.zeros((cluster0.shape[0], 1), dtype=int)
label1 = np.ones((cluster1.shape[0], 1), dtype=int)

# Append cluster labels to cluster data
cluster0 = np.append(cluster0, label0, axis=1)
cluster1 = np.append(cluster1, label1, axis=1)

# Merge both clusters and sort by ID (first column)
labeled_data = np.concatenate([cluster0, cluster1])
labeled_data = np.array(sorted(labeled_data, key=lambda x: int(x[0])))

# Load true class labels
diabetes = arff_to_ndarray("./diabetes.arff")
encoder = LabelEncoder()
labels_true = encoder.fit_transform(diabetes[1].flatten())
labels_pred = labeled_data[:, -1].astype(int)  # predicted labels

# Compute NMI
nmi = normalized_mutual_info_score(labels_true, labels_pred)
print(nmi)
