from typing import Tuple
import math

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Returns mu and sigma corresponding to a Binomial(n,p)"""
    mu=n * p
    sigma = math.sqrt(n * p * (1-p))
    return mu, sigma

from datascience.probability import normal_cdf

# Whenever a random variable follows a normal distribution, we use
# normal_cdf to figure out the probability that its realized value
# lies within or outside a particular interval.
# The normal cdf _is_ the probability the variable is below a threshold

normal_probability_below=normal_cdf

# It's above the threshold if it's not below the threshold
def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is greater than lo. """
    return 1 - normal_cdf(lo, mu, sigma)

# It's between if it's less than hi, but not less than lo
def normal_probability_between(lo: float, 
                               hi: float, 
			       mu: float = 0, 
			       sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is between lo and high"""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

#It's outside if it's not between
def normal_probability_outside(lo: float,
                               hi: float,
			       mu: float = 0,
			       sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is not between lo and high"""
    return 1 - normal_probability_between(lo, hi, mu, sigma)


from datascience.probability import inverse_normal_cdf

def normal_upper_bound(probability: float,
                       mu: float = 0,
		       sigma: float = 1) -> float:
    """ Returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float = 0,
		       sigma: float = 1) -> float:
    """ Returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
                       mu: float = 0,
		       sigma: float = 1) -> Tuple[float, float]:
    """ Returns the symmetric (about the mean) bounds that
        contain the specified probability"""
    tail_probability = (1 - probability) / 2
    
    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
assert mu_0 == 500
assert 15.7 < sigma_0 < 15.9

# type 1 error ("false positive"), in which we reject H0 even though
# it's true. In other words, if H0 is true, approximately 19 times
# out of 20, this test will give the correct result. 
# (469, 531)
low_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)
print("5% low_bound, upper_bound:",low_bound, upper_bound)
assert 468.9 < low_bound < 469.1
assert 530.5 < upper_bound < 531.5

#low_bound, upper_bound = normal_two_sided_bounds(0.01, mu_0, sigma_0)
#print("1% low_bound, upper_bound:",low_bound, upper_bound)
#assert 499.8 < low_bound < 500
#assert 500 < upper_bound < 500.2

# actual mu and sigma based on p=0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)
print(" p=0.55, mu_1, sigma_1", mu_1, sigma_1)

# a type 2 error ("false negative") means we fail to reject H0,
# which will happen when X is still in our original interval.
type_2_probability = normal_probability_between(low_bound, upper_bound, mu_1, sigma_1)
power = 1 - type_2_probability
print("power = ",power)

upper_bound = normal_upper_bound(0.95, mu_0, sigma_0)
print("upper_bound:", upper_bound)

type_2_probability = normal_probability_below(upper_bound, mu_1, sigma_1)
power = 1 - type_2_probability
print("power=", power)

# p-values
def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """How likely are we to see a value at least as extreme as x
    (in either direction) if our values are from an N(mu, sigma)?"""
    if x >= mu:
        # x is greater than the mean, so the tail is everything greather than x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # x is less than the mean, so the tail is everything less than x
        return 2 * normal_probability_below(x, mu, sigma)

result = two_sided_p_value(529.5, mu_0, sigma_0)    #529.5 rather than 530
print("to see 530 heads, p=", result)  #0.062

# one way to convince yourself this is sensible estimate is with a simulation
import random
extreme_value_count = 0

for _ in range(1000):
    num_heads = sum(1 if random.random() < 0.5 else 0
                    for _ in range(1000))
    if num_heads >= 530 or num_heads <= 470:
        extreme_value_count +=1

print("extreme value (>=530 or <=470):", extreme_value_count)
assert 59 < extreme_value_count < 65, f"{extreme_value_count}"

result = two_sided_p_value(531.5, mu_0, sigma_0)
two_sided_p_value(529.5, mu_0, sigma_0)
print("to see 532 heads:", result)
if result < 0.05:
    print("reject h0")

upper_p_value = normal_probability_above
loser_p_value = normal_probability_below

result = upper_p_value(524.5, mu_0, sigma_0)
print("to see 525 heads:", result)
print("wouldn't reject h0") if result > 0.05 else print("reject h0")

result = upper_p_value(526.5, mu_0, sigma_0)
print("to see 527 heads:", result)
print("wouldn't reject h0") if result > 0.05 else print("reject h0")
print("\n\n\n")

# confidence intervals
# 525
p_hat = 525/1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)
print("mu, sigma=", mu, sigma)
result = normal_two_sided_bounds(0.95, mu, sigma)
print(result)
lo, hi= result
if lo < 0.5 and hi > 0.5:
    print("0.5 falls within confidence interval.")
else:
    print("0.5 doesn't lie in the confidence interval")
        
# 540
p_hat = 545/1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)
print("mu, sigma=", mu, sigma)
result = normal_two_sided_bounds(0.95, mu, sigma)
print(result)
lo, hi= result
if lo < 0.5 and hi > 0.5:
    print("0.5 falls within confidence interval.")
else:
    print("0.5 doesn't lie in the confidence interval")

# p-hacking
# A procedure that erroneously rejects the h0 only 5% of the time
# will -by definition - 5% of the time erroneously reject the h0
from typing import List
def run_experiment() -> List[bool]:
    """Flip a fair coin 1000 times, True=head, False=tail"""
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment: List[bool]) -> bool:
    """Using the 5% significance level"""
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

random.seed(0)
experiments = [run_experiment() for _ in range(1000)]
num_rejections = len([experiment
                      for experiment in experiments
                      if reject_fairness(experiment)])

print("# of rejections:", num_rejections)
assert num_rejections == 46

# A/B Testing
def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1-p) / N)
    return p, sigma

def a_b_test_statistics(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt( sigma_A ** 2 + sigma_B ** 2)

z= a_b_test_statistics(1000,200, 1000, 180)
print("result of (1000,200) vs (1000, 180):", z)
result = two_sided_p_value(z)
print(result)
print("\n")

z= a_b_test_statistics(1000,200, 1000, 150)
print("result of (1000,200) vs (1000, 150):", z)
result = two_sided_p_value(z)
print(result)

# Bayesian Inference
def B(alpha: float, beta: float) -> float:
    """ A normalizing constant so that the total probability is 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:
        return 0
    return x ** (alpha -1) * (1 - x) ** (beta -1) / B(alpha, beta)

# this distribution centers its weight at alpha / (alpha + beta)
# and the larger alpha and beta are, the "tighter" the distribution is
# if alpha and beta are both 1, just the uniform distribution
# (centered at 0.5, very dispersed). 

