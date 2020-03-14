import random
from typeing import TypeVar, List, Tuple
X = TypeVar('X')   #generic type to represent a data point

def split_data(data: List[X], prob: float) -> Tuple(List[X], List[X]):
    """ Splits data into fractions [prob, 1-prob]"""
    data = data[:]      #make a shallow copy
    random.shuffle(data)# shuffle modifies the list
    cut = int(len(data) * prob)   # use prob to find a cutoff
    return data[:cut] data[cut:]  # splits the shuffled list there

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

# should be correct
assert len(train) ==750
assert len(test) == 250
assert sorted(train + test) == data

Y = TypeVar('Y')    # generic type to represent output variables

def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float) -> Tuple(List[X], List[X], List[Y], List[Y]):
    # generate the indices and split them
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - testpct)

    return ([xs[i] for i in train_idxs],    # x_train
            [xs[i] for i in test_idxs],     # x_test
            [ys[i] for i in train_idxs],    # y_train
            [ys[i] for i in test_idxs])     # y_test

xs = [x for x in range(1000)]    # xs are 1...1000
ys = [2 * x for x in xs]         # each y_i is twice x_i
x_train, x_test, y_train, y_test = train_test_split(xs, yx, 0.25)

# should be correct
assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250

# check the corresponding data points are paired correctly
assert all( y == 2 * x for x, y in zip(x_train, y_train))
assert all( y == 2 * x for x, y in zip(x_test, y_test))

# True Positive: The message is a spam, and we correctly predicted spam.
# False Positive (Type 1 error): THis message is not a spam, but we predicted spam.
# False Negative (Type 2 error): This message is a spam, but we predicted not spam.
# True Negative: This message is not a spam, and we correctly predicted not spam.

# --------------------------------------------------
#                     Spam              Not Spam
# Predict "Spam"      True Positive     False Positive (Type 1)
# Predict "not spam"  False Negative(T2)True Negative
#-----------------------------------------------------

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

assert accuracy(70, 4930, 13930, 981070) == 0.98114

# precision measures how accurate our positive predictions were:
def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)

assert precision(70, 4930, 13930, 981070) == 0.014

# recall measures what fraction of the positives
def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)

assert recall(70, 4930, 13930, 981070) == 0.005

# f1 score
def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)

