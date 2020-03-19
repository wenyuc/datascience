# uniform pdf
# Probability Density Function(PDF) that the probability of seeing a value
# in a certain interval equals the integral of the density function over the interval
def uniform_pdf(x:float) -> float:
    return 1 if 0 <= x < 1 else 0

# normal cdf
# Cumulative distribution function(CDF) that gives the probability that a random
# variable is less than or equals to a certain value
def uniform_cdf(x: float) -> float:
    """ Returns the probability that a uniform random variable is <= x"""
    if x < 0: return 0     # uniform random is never less than 0
    elif x< 1: return x    # e.g. P(X <= 0.4) = 0.4
    else:  return 1        # uniform random is always less than 1

#4 normal distribution pdf, cdf and inverse_cdf.

import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return(math.exp(-(x-mu) ** 2 / 2 /sigma ** 2) / (SQRT_TWO_PI * sigma))

import matplotlib.pyplot as plt
xs = [x/10.0 for x in range(-50,50)]
plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
plt.plot(xs, [normal_pdf(x, mu=-1) for x in xs], '-.', label='mu=-1, sigma=1')
plt.legend()
plt.title("Vairous Normal pdfs")
#plt.show()
plt.savefig("im/Various_Normal_Pdfs.png")
plt.close()

# x mapping to p
def normal_cdf(x:float, mu: float = 0, sigma: float=1) -> float:
    return(1 + math.erf((x-mu) / math.sqrt(2) /sigma)) /2

xs = [x/10.0 for x in range(-50,50)]
plt.plot(xs, [normal_cdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
plt.plot(xs, [normal_cdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
plt.plot(xs, [normal_cdf(x, mu=-1) for x in xs], '-.', label='mu=-1, sigma=1')
plt.legend(loc = 4)    #bottom right
plt.title("Vairous Normal cdfs")
#plt.show()
plt.savefig("im/Various_Normal_Cdfs.png")
plt.close()

# p mapping to x
def inverse_normal_cdf(p:float,
                       mu: float = 0,
		       sigma: float = 1,
		       tolerance: float = 0.00001) -> float:
    """Find approximate inverse using binary search"""

    # if not standard, compute standard and rescale
    if mu != 0 or sigma !=1:
        # if Z is a standard normal distribution(u = 0, sigma = 1),
        # X = sigma * Z + u is also normal with mean u and standard
        # variance = sigma
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10.0     #normal_cdf(-10) is (very close to) 0
    hi_z = 10.0       #normal_cdf(10) is (very close to) 1

    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2    #consider the midpoint
        mid_p = normal_cdf(mid_z)       # compute the CDF's value
        if mid_p < p:
            low_z = mid_z               # midpoint too low, search above it
        else:
            hi_z = mid_z                # midpoint to high, search below it

    return mid_z

# The Central Limit Theorem, Bernolli distribution
import random
def bernoulli_trial(p: float) -> int:
    """Returns 1 with probability p and 0 with probability 1-p"""
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    """A Binomial(n,p) random variable is simply the sum of 
    n independent Bernoulli(p) random variable"""
    """ Returns the sum of n bernoulli(p) trials"""
    return sum(bernoulli_trial(p) for _ in range(n))

# The mean of a Bernoulli(p) variable is p, and its standard deviation is
# sqrt(p*(1-p)). The CENTRAL LIMIT THEOREM says that as n gets large, 
# a Binomial(n,p) variable is approximately a normal random variable with
# mean mu = n*p and standard deviation seta=sqrt(n*p*(1-p)).
# The following function plot both.

from collections import Counter
from matplotlib import pyplot as plt

def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """Picks points from a Binomial(n,p) and plots their histogram"""

    data = [binomial(n,p) for _ in range(num_points)]
    #print(data)

    histogram= Counter(data)
    #print(histogram)

    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
	    0.8, color='0.75')
    
    mu = n * p
    sigma = math.sqrt(n * p * (1-p))

    # use a line chart to show the normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]
    plt.plot(xs, ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    #plt.show()
    plt.savefig("im/Binomial_Normal_Approximation.png")
    plt.close()
    
binomial_histogram(0.75,100,10000)

def main():
    # conditional probability
    import enum, random

    # Enum is a typed set of enumerated values.
    class Kid(enum.Enum):
        BOY = 0
        GIRL = 1

    def random_kid() -> Kid:
        return random.choice([Kid.BOY, Kid.GIRL])

    both_girls = 0
    older_girl = 0
    either_girl = 0

    for _ in range(10000):
        older = random_kid()
        younger = random_kid()

        if older == Kid.GIRL:
            older_girl += 1
        if older == Kid.GIRL and younger == Kid.GIRL:
            both_girls += 1
        if older == Kid.GIRL or younger == Kid.GIRL:
            either_girl += 1

    print("P(both | older):", both_girls / older_girl)
    print("P(both | either):", both_girls / either_girl)
    
if __name__ == "__main__": main()
