#!/usr/bin/env python3

import numpy as np
import numpy.linalg as linalg

import random as rnd

def kmeanspp(DB, k):
    seeds = [DB[np.random.randint(len(DB))]]

    while len(seeds) < k:
        Di_sq = [min([linalg.norm(pt - seed, 2)**2 for seed in seeds]) for pt in DB]
        sum_Di_sq = sum(Di_sq)
        pr = list(map(lambda x: x / sum_Di_sq, Di_sq))

        seeds.append(DB[np.random.choice(len(DB), p=pr)])

    return seeds

def random(DB, k):
    return list(DB[rnd.sample(range(len(DB)), k)])
