#!/usr/bin/env python3

import numpy as np
import numpy.linalg as linalg
import sys
import itertools
import seeding_strategy
import logging
from math import exp, log

logging.basicConfig(stream=sys.stderr, level=logging.NOTSET) # set to warning, error or critical to silence

def pdist(x, y, vectors):
    return linalg.norm(vectors.T @ x.T - vectors.T @ y.T, 2)

def cluster_energy(cluster, centroid, vectors):
    return np.sum([pdist(p, centroid, vectors)**2 for p in cluster])/len(cluster)

def orclus(DB, k, l, alpha=0.5, k0=None):
    kc = 5*k if k0 is None else k0 # current number of seeds
    lc = DB.shape[1] # current dimensionality associated with each seed

    if l > lc:
        print("target dimensionality must be lower-equal than input dimensionality")
        return None

    if kc <= k or kc >= DB.shape[0]:
        print(f"default k0 does not work. specify different k0. default k0 = {kc}, but conflict with k = {k} or n = {n}")
        return None

    if k >= DB.shape[0]:
        print("k is larger or equal n but should be k << n")
        return None
    
    # choose initial seeds with K-means++
    seeds = seeding_strategy.kmeanspp(DB, kc)
    # initially, the vectors defining defining the subspaces are the original axis-system
    vectors = [np.eye(lc)]*kc

    beta = exp((-log(lc/l)*log(1/alpha))/log(kc/k))

    while kc > k:
        logging.info("assigning points to centroids")
        # find partitioning induced by the seeds of the clusters
        seeds, clusters = assign(DB, seeds, vectors)
        k_new = int(max(k, kc*alpha)); l_new = int(max(l, lc*beta))

        logging.info("finding cluster vectors")
        # determine current subspace associated with each cluster
        vectors = [find_vectors(cluster, lc) for cluster in clusters]

        logging.info("merging clusters")
        # reduce number of seeds and dimensionality associated with each seed
        seeds, clusters, vectors = merge(seeds, clusters, k_new, l_new)

        kc = k_new; lc = l_new
        logging.info(f"new kc = {kc}")
        logging.info(f"new lc = {lc}")

    seeds, clusters = assign(DB, seeds, vectors)

    return (clusters, seeds, vectors)

def find_vectors(cluster, q):
    if len(cluster) <= 1: return np.eye(q) # cannot decompose

    _, v = linalg.eigh(np.cov(np.vstack(cluster), rowvar=False))
    return v[:, 0:q]

def assign(DB, seeds, vectors):
    clusters = [[] for _ in range(len(seeds))]
    for p in DB:
        # distances of point p to seeds in the given subspace vectors
        dist = []
        for s, v in zip(seeds, vectors):
            dist.append(pdist(p, s, v))
        # add p to the cluster associated with the closest seed
        clusters[np.argmin(dist)].append(p)
    
    # set seeds to new cluster centers
    for i in range(0, len(seeds)):
        seeds[i] = np.sum(clusters[i], axis=0)/len(clusters[i])

    return (seeds, clusters)

def merge(seeds, clusters, k_new, l_new):
    merged_clusters = []

    for a, b in itertools.combinations(enumerate(clusters), 2):
        merged_cluster = a[1] + b[1]
        
        # define vectors by eigenvectors for l_new smallest eigenvalues
        vectors = find_vectors(merged_cluster, l_new)
        # centroid of merged clusters
        centroid = np.sum(merged_cluster, axis=0)/len(merged_cluster)
        # compute projected energy of merged clusters in subspace vectors
        energy = cluster_energy(merged_cluster, centroid, vectors)

        merged_clusters.append((a[0], b[0], centroid, energy))

    while len(seeds) > k_new:
        # merge cluster pair i, j with smallest energy value
        i_, j_, centroid, energy = min(merged_clusters, key=lambda t: t[3])
     
        seeds[i_] = centroid
        clusters[i_] += clusters[j_]
        
        del seeds[j_]
        del clusters[j_]

        merged_clusters = [t for t in merged_clusters if t[0] != j_ and t[1] != j_]
        #renumber the seeds/clusters
        for idx, (i, j, centroid, energy) in enumerate(merged_clusters):
            if i > j_ and j > j_:
                merged_clusters[idx] = (i-1, j-1, centroid, energy)
            elif i > j_:
                merged_clusters[idx] = (i-1, j,   centroid, energy)
            elif j > j_:
                merged_clusters[idx] = (i  , j-1, centroid, energy)
        
        #recompute energy for new cluster
        for idx, (i, j, centroid, energy) in enumerate(merged_clusters):
            if i != i_: continue

            merged_cluster = clusters[i] + clusters[j]
            vectors = find_vectors(merged_cluster, l_new)
            centroid = np.sum(merged_cluster, axis=0)/len(merged_cluster)
            energy = cluster_energy(merged_cluster, centroid, vectors)

            merged_clusters[idx] = ((i, j, centroid, energy))

    return (seeds, clusters, [find_vectors(cluster, l_new) for cluster in clusters])

def predict(DB, seeds, vectors):
    clustering = []
    for p in DB:
        dist = []
        for s, v in zip(seeds, vectors):
            dist.append(pdist(p, s, v))

        clustering.append(np.argmin(dist))

    return clustering
