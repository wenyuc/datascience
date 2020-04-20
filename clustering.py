#!/usr/bin/env python
# coding: utf-8

# In[6]:


# helper functions
from datascience.linear_algebra import Vector

# to measure how many corrdinates two vectors differ in
def num_differences(v1: Vector, v2: Vector) -> int:
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])


assert num_differences([3,5,7], [5,3,7]) == 2
assert num_differences([1,3,5,7], [1,3,5,7]) == 0

# given some vectors and their assignments to clusters,
# computes the means of the clusters
from typing import List
from datascience.linear_algebra import vector_mean

def cluster_means(k: int, 
                 inputs: List[Vector],
                 assignments: List[int]) -> List[Vector]:
    # clusters[i] contains the inputs whose assignment is i
    clusters = [[] for i in range(k)]
    for input, assignment in zip(inputs, assignments):
        clusters[assignment].append(input)
    
    return [vector_mean(cluster) if cluster else random.choice(inputs)
           for cluster in clusters]


inputs: List[List[float]] = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],
                             [-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],
                             [-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

import random
assignments = [random.randrange(3) for _ in inputs]
print(assignments)

cl = cluster_means(3, inputs, assignments)
print(cl)

# In[7]:


import itertools
import random
import tqdm
from datascience.linear_algebra import squared_distance

class KMeans:
    def __init__(self, k: int) -> None:
        self.k = k           # number of clusters
        self.means = None
    
    def classify(self, input: Vector) -> int:
        """Returns the index of the cluster closest to the input"""
        return min(range(self.k),
                  key =lambda i: squared_distance(input, self.means[i]))
    
    def train(self, inputs: List[Vector]) -> None:
        # Starts with random assignments
        assignments = [random.randrange(self.k) for _ in inputs]
        print(assignments)
 
        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # Computes means and find new assignments
                self.means = cluster_means(self.k, inputs, assignments)
                new_assignments = [self.classify(input) for input in inputs]
                
                # Checks how many assignments change and if we're done
                num_changed = num_differences(assignments, new_assignments)
                
                if num_changed == 0:
                    return
                
                # otherwise keep the new assignments, and computes new means
                assignments = new_assignments
                self.means = cluster_means(self.k, inputs, assignments)
                t.set_description(f"changed: {num_changed} / {len(inputs)}")     


# In[8]:


def main():
    inputs: List[List[float]] = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],
                                 [-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],
                                 [-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
    
    random.seed(10)
    clusterer = KMeans(k = 3)
    clusterer.train(inputs)
    means = sorted(clusterer.means)
    
    assert len(means) == 3
    
    # Check that the means are close to what we expect.
    assert squared_distance(means[0], [-44, 5]) < 1
    assert squared_distance(means[1], [-16, -10]) < 1
    assert squared_distance(means[2], [18, 20]) < 1
    
    random.seed(0)
    clusterer = KMeans(k=2)
    clusterer.train(inputs)
    means = sorted(clusterer.means)

    assert len(means) == 2
    assert squared_distance(means[0], [-26, -5]) < 1
    assert squared_distance(means[1], [18, 20]) < 1

# if __name__ == "__main__": main()
        


# In[ ]:




