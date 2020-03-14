from typing import List
from collections import Counter

# count votes
def raw_majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    # print(votes.most_common(1)[0])
    return winner

assert raw_majority_vote(['abc', 'abc','def', 'kpm','abc', 'kpm', 'dsf']) == 'abc'

# break tie
def majority_vote(labels: List[str]) -> str:
    """Assume the labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner       # unique winner, so return it
    else:
        return majority_vote(labels[:-1])   # try again without the farthest

# Tie, so look at first 4, then 'de'

assert majority_vote(['en', 'de','jp', 'de', 'jp','en', 'de','jp']) == 'de'

# to create a classifier
from typing import NamedTuple
from mymath.linear_algebra import Vector, distance

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int,
                 labeled_points: List[LabeledPoint],
                 new_point: Vector) -> str:
    """Order the labeled points from nearest to farthest"""
    by_distance = sorted(labeled_points,
                         key = lambda lp: distance(lp.point, new_point))

    # find the labels for the k closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    # and let them vote.
    return majority_vote(k_nearest_labels)

import requests

data = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

with open('iris.data', 'w') as f:
    f.write(data.text)


