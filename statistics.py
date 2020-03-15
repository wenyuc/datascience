num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

# 1 histogram a list
from collections import Counter
from matplotlib import pyplot as plt

friend_counts = Counter(num_friends)
print(friend_counts)
max_key = max(friend_counts.keys())
max_value = max(friend_counts.values())
print("max_key, max_value:", max_key, max_value)

num_points = len(num_friends)

# first way to get largest and smallest value
largest_value = max(num_friends)
smallest_value = min(num_friends)
print("1: largest={0} smallest={1}", largest_value, smallest_value)

# second way to get largest and smallest value
sorted_values = sorted(num_friends)
largest_value2 = sorted_values[-1]
smallest_value2 = sorted_values[0]
second_largest_value = sorted_values[-2]
print("2: largest={0} second_largest= {1} smallest={2}", largest_value2,
      second_largest_value,smallest_value2)

xs = range(int(largest_value)+1)                      #largest value is 100
ys = [friend_counts[x] for x in xs]  # height is just # of friends
plt.bar(xs, ys)
plt.axis([0,101, 0,25])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
#plt.show()
plt.savefig("im/friend_counts.png")

#2 mean, median, quantile, mode, data_range
from typing import List

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

print("mean # of friends:",mean(num_friends))

# the underscores indicate that these are "private" functions, as they're
# indicated to be called by other functions (i.e., our median function),
# but not by other people using our statistics library
def _median_odd(xs: List[float]) -> float:
    """If len(xs) is odd, the median is the middle element"""
    return sorted(xs)[len(xs) // 2]

def _median_even(xs: List[float]) -> float:
    """If len(xs) is even, the median is the average of the middle 2 elements"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2    #e.g. length = 4 => hi_midpoint=2
    return ((sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2)

def median(v: List[float]) -> float:
    """Find the median of a list of numbers."""
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

print("median([1,10,2,9,5])=")
print(median([1,10,2,9,5]))
print("median([1,9,2,10])=")
print(median([1,9,2,10]))
assert median([1,9,2,10]) == (2+9) / 2

print("median of num_friends=", median(num_friends))

def quantile(xs: List[float], p: float) -> float:
    """Returns the pth-percentile value in x"""
    p_index=int(p * len(xs))
    return sorted(xs)[p_index]

print("10% of num_friends:",quantile(num_friends, 0.10))
print("25% of num_friends:",quantile(num_friends, 0.25))
print("75% of num_friends:",quantile(num_friends, 0.75))
print("90% of num_friends:",quantile(num_friends, 0.90))

def mode(x: List[float]) -> List[float]:
    """Returns a list, since there might be more than one mode"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]

print("mode of num_friends:", mode(num_friends))
assert set(mode(num_friends)) == {1,6}

def data_range(xs: List[float]) -> float:
    """Returns the difference between max and min"""
    return max(xs) - min(xs)

print("range of num_friends:",data_range(num_friends))
assert data_range(num_friends) == 99

#3 variance, standard deviation, interquartile_range
from datascience.linear_algebra import sum_of_squares

def de_mean(xs: List[float]) -> List[float]:
    """Translate xs by subtracting its mean (so the result has mean 0)"""
    x_bar=mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
    """Almost the average squared deviation from the mean"""
    assert len(xs) >= 2, "variance requires at least two elements"

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)   # why (n-1), not n? https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation

print("variance of num_friends:", variance(num_friends))
assert 81.54 < variance(num_friends) < 81.55

import math
def standard_deviation(xs: List[float]) -> float:
    """the standard deviation is the square root of the variance"""
    return math.sqrt(variance(xs))

print("standard deviation of num_friends:", standard_deviation(num_friends))
assert 9.02 < standard_deviation(num_friends) < 9.04

# in order to avoide the outlier problem, computes the difference
# between the 75th percentile value and 25th percentile value
def interquartile_range(xs: List[float]) -> float:
    """ Returns the difference between the 75th percentile value and 25th percentile value"""
    return quantile(xs, 0.75) - quantile(xs, 0.25)

print("diff btw 75-th and 25-th:", interquartile_range(num_friends))

#4 covariance, correlation, take away outlier
daily_minutes = [1,68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]

daily_hours = [dm / 60 for dm in daily_minutes]

from datascience.linear_algebra import dot
def covariance(xs: List[float], ys: List[float]) -> float:
    """ Covariance measures hwo two variables vary in tandem from their means"""
    assert len(xs) == len(ys), "xs and ys must have the same length"

    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1 )

print("covariance btw. num_friends and daily_minutes:", covariance(num_friends, daily_minutes))
assert 22.42 < covariance(num_friends, daily_minutes) < 22.43
assert 22.42 / 60 < covariance(num_friends, daily_hours) < 22.43 / 60

def correlation(xs: List[float], ys: List[float]) -> float:
    """ measures how much xs and ys vary in tandem about their means"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)

    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0      #if no variation, correlation is zero

print("correlation of num_friends, daily_minutes:", correlation(num_friends, daily_minutes))
assert 0.24< correlation(num_friends, daily_minutes) < 0.25    # always btw. -1(anti-correlation) and 1(perfect correlation)
assert 0.24< correlation(num_friends, daily_hours) < 0.25   

outlier = num_friends.index(100)
print("outlier:", outlier)
num_friends_good = [x
                    for i, x in enumerate(num_friends)
                    if i != outlier]

daily_minutes_good = [x
                      for i, x in enumerate(daily_minutes)
                      if i != outlier]

daily_hours_good = [dm / 60 for dm in daily_minutes_good]

print("correlation of num_friends_good, daily_minutes_good:", correlation(num_friends_good, daily_minutes_good))
assert 0.57 < correlation(num_friends_good, daily_minutes_good) < 0.58    # always btw. -1(anti-correlation) and 1(perfect correlation)
assert 0.57 < correlation(num_friends_good, daily_hours_good) < 0.58
