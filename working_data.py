from typing import List, Dict
from collections import Counter
import math

def bucketize(point: float, bucket_size: float) -> float:
    """Floor the point to the next lower multiple of bucket_size"""
    return bucket_size * math.floor(point / bucket_size)

from datascience.linear_algebra import Vector
import matplotlib.pyplot as plt

def make_histogram(points: Vector, bucket_size: float) -> Dict[float, int]:
    """Buckets the points and counts how many in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points: Vector, bucket_size: float, title: str = "") -> None:
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width = bucket_size)
    plt.title(title)
    #plt.show()
    
import random
from datascience.probability import inverse_normal_cdf

random.seed(0)

# uniform between -100 and 100
uniform = [200 * random.random() - 100 for _ in range(10000)]

# normal distribution with mean 0, sd 57
normal = [57 * inverse_normal_cdf(random.random()) for _ in range(10000)]

plot_histogram(uniform, 10, "Uniform Histogram")
plot_histogram(normal, 10, "Normal Histogram")

def random_normal() -> float:
    """Returns a random draw from a standard normal distribution"""
    return inverse_normal_cdf(random.random())

xs = [ random_normal() for _ in range(1000)]
ys1 = [ x + random_normal() / 2 for x in xs]
ys2 = [ -x + random_normal() / 2 for x in xs]

plot_histogram(ys1, 10, "ys1 Histogram")
plot_histogram(ys2, 10, "ys2 Histogram")

# joint distribution of ys1,ys2 with xs
plt.scatter(xs, ys1, marker=".", color = "black", label = "ys1")
plt.scatter(xs, ys2, marker=".", color = "gray", label = "ys2")
plt.xlabel("xs")
plt.ylabel("ys")
plt.legend(loc=9)
plt.title("Very Different Joint Distrbution")
#plt.show()

# to check the correlation of ys1, ys2 with xs
from datascience.statistics import correlation
print(correlation(xs, ys1))
print(correlation(xs, ys2))

from datascience.linear_algebra import Matrix, make_matrix

def correlation_matrix(data: List[Vector]) -> Matrix: 
    """Returns the len(data) * len(data) matrix whose (i,j)-th
       is the correlation between data[i] and data[j]"""
    def correlation_ij(i: int, j: int) -> float:
        return correlation(data[i], data[j])\

    return make_matrix(len(data), len(data), correlation_ij)

# Just some random data to show off correlation scatterplots
num_points = 100

def random_row() -> List[float]:
   row = [0.0, 0, 0, 0]
   row[0] = random_normal()
   row[1] = -5 * row[0] + random_normal()
   row[2] = row[0] + row[1] + 5 * random_normal()
   row[3] = 6 if row[2] > -2 else 0
   return row

random.seed(0)
# each row has 4 points, but really we want the columns
corr_rows = [random_row() for _ in range(num_points)]
corr_data = [list(col) for col in zip(*corr_rows)]

# corr_data is a list of four 100-d vectors
num_vectors = len(corr_data)
fig, ax = plt.subplots(num_vectors, num_vectors)

for i in range(num_vectors):
    for j in range(num_vectors):

        # scatter column_j on x-axis vs. column_i on the y-axis
        if i != j:
            ax[i][j].scatter(corr_data[j], corr_data[i])
        else:   # unless i == j, in which case show the series name
            ax[i][j].annotate("series " + str(i), (0.5, 0.5),
                              xycoords = "axes fraction",
                              ha = "center", va = "center")

        #Then hide axis labels except left and bottom charts
        if i < num_vectors - 1: ax[i][j].xaxis.set_visible(False)
        if j > 0: ax[i][j].yaxis.set_visible(False)

# Fix the bottom-right and top-left axis labels, which are wrong
# because their charts only have text in them
ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())
#plt.show()

# Namedtuple, immutable
from collections import namedtuple
import datetime

StockPrice = namedtuple('StockPrice', ['symbol', 'date', 'closing_price'])
price = StockPrice('MSFT', datetime.date(2018, 12,14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03

from typing import NamedTuple

class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """ It is a class, we can add methods to"""
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMAZ', 'APPL']

price = StockPrice('MSFT', datetime.date(2018, 12,14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03
assert price.is_high_tech()

# dataclass, like NamedTuple, but dataclass is mutable
from dataclasses import dataclass

@dataclass
class StockPrice2:
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """ It is a class, we can add methods to"""
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMAZ', 'APPL']

price2 = StockPrice2('MSFT', datetime.date(2018, 12,14), 106)

assert price2.symbol == 'MSFT'
assert price2.closing_price == 106
assert price2.is_high_tech()

# stock split
price2.closing_price /= 2
assert price2.closing_price == 53

# parse a list
from typing import Optional
import re
from dateutil.parser import parse

def try_parse_row(row: List[str]) -> Optional[StockPrice]:
    symbol, date_, closing_price_ = row

    # Stock name should be all capital letters
    if not re.match(r"^[A-Z]+$", symbol):
        return None

    try:
        date = parse(date_).date()
    except ValueError:
        return None

    try:
        closing_price = float(closing_price_)
    except ValueError:
        return None

    return StockPrice(symbol, date, closing_price)

# Test
assert try_parse_row(["MSFT0", "2018-12-14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12--14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12-14", "x"]) is None

assert try_parse_row(["MSFT", "2018-12-14", "106.03"]) == price

import csv

data: List[StockPrice] = []

if 0:
    with open("comma_delimited_stock_prices.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)
            maybe_stock = try_parse_row(row)
            if maybe_stock is None:
                print(f"skipping invalid row: {row}")
            else:
                data.append(maybe_stock)

    max_appl_price = max(stock_price.closing_price
                         for stock_price in data
                         if stock_price.symbol == 'AAPL')
    print("max apple price:", max_appl_price)

with open("stocks.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = [[row['Symbol'], row['Date'], row['Close']]
            for row in reader]

# skip header
maybe_data = [try_parse_row(row) for row in rows]

# make sure they all loaded successfully
assert maybe_data
assert all(sp is not None for sp in maybe_data)
data = [sp for sp in maybe_data if sp is not None]

# to get each stock's highest price
from collections import defaultdict

max_prices: Dict[str, float] = defaultdict(lambda: float('-inf'))

for sp in data:
    symbol, closing_price = sp.symbol, sp.closing_price
    if closing_price > max_prices[symbol]:
        max_prices[symbol] = closing_price

print(max_prices)

# to group the prices by symbol
from typing import List
from collections import defaultdict

# collect the prices by symbol
prices: Dict[str, List[StockPrice]] = defaultdict(list)

for sp in data:
    prices[sp.symbol].append(sp)

#print("symbol group:", prices)

# order the prices by date
prices = {symbol: sorted(symbol_prices)
          for symbol, symbol_prices in prices.items()}
#print("symbol group sorted by date:", prices)

def pct_change(yesterday: StockPrice, today: StockPrice) -> float:
    return today.closing_price / yesterday.closing_price - 1

class DailyChange(NamedTuple):
    symbol: str
    date: datetime.date
    pct_change: float

def day_over_day_change(prices: List[StockPrice]) -> List[DailyChange]:
    """Assume prices are for one stock and are in order"""
    return [DailyChange(symbol = today.symbol,
                        date = today.date,
                        pct_change = pct_change(yesterday, today))
            for yesterday, today in zip(prices, prices[1:])]

# to collect them all
all_changes = [change
               for symbol_prices in prices.values()
               for change in day_over_day_change(symbol_prices)]

max_change = max(all_changes, key = lambda change: change.pct_change)
print("max change: ", max_change)
# see e.g. http://news.cnet.com/2100-1001-202143.html
assert max_change.symbol == 'AAPL'
assert max_change.date == datetime.date(1997, 8, 6)
assert 0.33 < max_change.pct_change < 0.34
min_change = min(all_changes, key=lambda change: change.pct_change)
# see e.g. http://money.cnn.com/2000/09/29/markets/techwrap/
assert min_change.symbol == 'AAPL'
assert min_change.date == datetime.date(2000, 9, 29)
assert -0.52 < min_change.pct_change < -0.51

# to find which month to invest
changes_by_month: List[DailyChange] = {month: [] for month in range(1,13)}

for change in all_changes:
    changes_by_month[change.date.month].append(change)

avg_daily_change = {
    month: sum(change.pct_change for change in changes) / len(changes)
               for month, changes in changes_by_month.items()
}

assert avg_daily_change[10] == max(avg_daily_change.values())

from typing import Tuple
from datascience.linear_algebra import vector_mean
from datascience.statistics import standard_deviation

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """returns each mean and sd for each item"""
    dim = len(data[0])

    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]

    return means, stdevs

vectors = [[-3, -1, 1], [-1, 0, 1], [1,1,1]]
means, stdevs = scale(vectors)
assert means == [-1, 0, 1]
assert stdevs == [2, 1, 0]

# to create a new dataset
def rescale(data: List[Vector]) -> List[Vector]:
    """Rescale the input data so that each item has mean 0 and stdev 0. """
    dim = len(data[0])
    means, stdevs = scale(data)

    # make a copy of each vector
    rescaled = [v[:] for v in data]

    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]

    return rescaled

means, stdevs = scale(rescale(vectors))
assert means == [0,0,1]
assert stdevs == [1,1,0]

import tqdm

def primes_up_to(n: int) -> List[int]:
    primes = [2]

    with tqdm.trange(3, n) as t:
        for i in t:
            # i is prime if no smaller prime divides it
            i_is_prime = not any(i % p == 0 for p in primes)
            if i_is_prime:
                primes.append(i)

my_primes = primes_up_to(1000)

from datascience.linear_algebra import subtract,magnitude,dot

def de_mean(data: List[Vector]) -> List[Vector]:
    """ Returns the data to have mean 0 in every dimension"""
    mean = vector_mean(data)
    return [subtract(vector, mean) for vector in data]

def direction(w: Vector) -> Vector:
    mag = magnitude(w)
    return [w_i / mag for w_i in w]

# compute the variance of our dataset in the direction determined by w
def directional_variance(data: List[Vector], w: Vector) -> float:
    """ Returns the variance of x in the direction of w"""
    w_dir = direction(w)
    return sum(dot(v, w_dir) ** 2 for v in data)

def directional_variance_gradient(data: List[Vector], w: Vector) -> Vector:
    """The gradient of directional variance with respect to w"""
    w_dir = direction(w)
    return [sum(2 * dot(v, w_dir) * v[i] for v in data)
            for i in range(len(w))]

from datascience.gradient_descent import gradient_step

def first_principal_component(data: List[Vector],
                              n: int = 100,
                              step_size: float = 0.1) -> Vector:
    # Start with a random guess
    guess = [1.0 for _ in data[0]]

    with tqdm.trange(n) as t:
        for _ in t:
            dv = directional_variance(data, guess)
            gradient = directional_variance_gradient(data, guess)
            guess = gradient_step(guess, gradient, step_size)
            t.set_description(f"dv: {dv:.3f}")

    return direction(guess)

from datascience.linear_algebra import scalar_multiply

def project(v: Vector, w: Vector) -> Vector:
    """return the projection of v onto the direction w"""
    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)

from datascience.linear_algebra import subtract

def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
    """projects v onto w and subtracts the result from v"""
    return subtract(v, project(v, w))

def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
    return [remove_projection_from_vector(v, w) for v in data]

def pca(data: List[Vector], num_components: int) -> List[Vector]:
    components: List[Vector] = []
    for _ in range(num_components):
        component = first_principal_component(data)
        components.append(component)
        data = remove_projection(data, component)

    return components

def transform_vector(v: Vector, components: List[Vector]) -> Vector:
    return [dot(v, w) for w in components]

def transform(data: List[Vector], components: List[Vector]) -> List[Vector]:
    return [transform_vector(v, components) for v in data]

