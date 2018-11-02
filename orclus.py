#!/usr/bin/env python3

from math import exp, log
import numpy as np
import numpy.linalg as linalg

from utils import arff_to_ndarray
from sys import exit

import pandas as pd

import itertools
import random

from pprint import pprint

def pdist(x, y, vectors):
    return linalg.norm(vectors.T*x.T - vectors.T*y.T)

def cluster_energy(cluster, centroid, vectors):
    return np.sum([pdist(p, centroid, vectors)**2 for p in cluster])/len(cluster)

def orclus(DB, k, l, alpha=0.5, k0=None):
    kc = 5*k if k0 is None else k0
    lc = DB.shape[1]

    if l >= lc:
        print("target dimensionality must be lower than input dimensionality")
        return None

    seeds = list(DB[random.sample(range(len(DB)), kc)]) # TODO: kmeans++
    vectors = [np.matrix(np.eye(lc))]*kc

    beta = exp((-log(lc/l)*log(1/alpha))/log(kc/k))

    while kc > k:
        print("assign")
        seeds, clusters = assign(DB, seeds, vectors)
        k_new = int(max(k, kc*beta)); l_new = int(max(l, lc*alpha))

        print("find vectors")
        vectors = [find_vectors(cluster, kc) for cluster in clusters]

        print("merge")
        seeds, clusters = merge(seeds, clusters, k_new, l_new)

        kc = k_new; lc = l_new
        print(f"new kc = {kc}")

    return clusters

def find_vectors(cluster, q):
    _, v = linalg.eigh(np.cov(np.vstack(cluster), rowvar=False))
    return np.matrix(v[:, 0:q])

def assign(DB, seeds, vectors):
    clusters = [[] for _ in range(len(seeds))]
    for p in DB:
        dist = []
        for s, v in zip(seeds, vectors):
            dist.append(pdist(p, s, v))

        clusters[np.argmin(dist)].append(p)

    for i in range(0, len(seeds)):
        seeds[i] = np.sum(clusters[i], axis=0)/len(clusters[i])

    return (seeds, clusters)

def merge(seeds, clusters, k_new, l_new):
    merged_clusters = []

    for a, b in itertools.combinations(enumerate(clusters), 2):
        merged_cluster = a[1] + b[1]

        vectors = find_vectors(merged_cluster, l_new)
        centroid = np.sum(merged_cluster, axis=0)/len(merged_cluster)
        energy = cluster_energy(merged_cluster, centroid, vectors)

        merged_clusters.append((a[0], b[0], centroid, energy))

    while len(seeds) > k_new:
        idx, (i_, j_, centroid, energy) = min(enumerate(merged_clusters), key=lambda t: t[1][3])

        seeds[i_] = centroid
        clusters[i_] += clusters[j_]

        del seeds[j_]
        del clusters[j_]

        merged_clusters = [t for t in merged_clusters if t[0] != j_ and t[1] != j_]

        for idx, (i, j, centroid, energy) in enumerate(merged_clusters):
            if i > j_ and j > j_:
                merged_clusters[idx] = (i-1, j-1, centroid, energy)
            elif i >= j_:
                merged_clusters[idx] = (i-1, j,   centroid, energy)
            elif j >= j_:
                merged_clusters[idx] = (i  , j-1, centroid, energy)

        for idx, (i, j, centroid, energy) in enumerate(merged_clusters):
            if i != i_: continue

            merged_cluster = clusters[i] + clusters[j]
            vectors = find_vectors(merged_cluster, l_new)
            centroid = np.sum(merged_cluster, axis=0)/len(merged_cluster)
            energy = cluster_energy(merged_cluster, centroid, vectors)

            merged_clusters[idx] = ((i, j, centroid, energy))

    return (seeds, clusters)


data, y = arff_to_ndarray("diabetes.arff")
clusters = orclus(np.matrix(data), 2, 3)
print(len(clusters))
