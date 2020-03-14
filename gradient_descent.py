from mymath.linear_algebra import Vector, dot

def sum_of_squares(v: Vector) -> float:
    """ Computes the sum of squared elements in v"""
    return dot(v,v)

from typing import Callable

# if f is a function of one variable, its derivative at a point x
# measures how f(x) changes when we make a very small change to x.
# The derivative is defined as the limit of the difference quotinents.
# as h approaches zero
def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h

def square( x: float) -> float:
    return x * x

# has the derivative:
def derivative(x: float) -> float:
    return 2 * x

xs = range(-21, 22)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h=0.001) for x in xs]

# plot to show they're basically the same
import matplotlib.pyplot as plt
plt.title("Actual Derivatives vs. Estimates")
plt.plot(xs, actuals, 'rx', label = 'Actual')
plt.plot(xs, estimates, 'b+', label = 'Estimate')
plt.legend(loc=9)
plt.show()
#plt.savefig("im/Actual_Derivatives_Estimates")

def partial_difference_quotient(f: Callable[[float], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """Returns the i-th partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0)     #add h to just the ith element of v
         for j, v_j in enumerate(v)]
    return (f(w) - f(v)) /h

def estimate_gradient(f: Callable[[float], float],
                      v: Vector,
                      h: float = 0.0001) -> Vector:
    return [partial_difference_quotient(f,v,i,h)
            for i in range(len(v))]

# using ghe gradient
import random
from mymath.linear_algebra import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """ Moves 'step_size' in the 'gradient' direction from v"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

# pick a random starting point
v = [random.uniform(-10, 10) for i in range(3)]
print(v)

for epoch in range(1000):
    grad = sum_of_squares_gradient(v)    # computes the gradient at v
    v = gradient_step(v, grad, -0.01)   # take a negative gradient step
    print(epoch, v)

print("distance:", distance(v, [0,0,0]))
assert distance(v, [0,0,0]) < 0.0001

# Using Gradient Descent to Fit Models

# x ranges from -50 to 49, y is always 20 * x + 5
inputs = [(x, 20 * x + 5) for x in range (-50, 49)]

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept      # The prediction of the model
    error = (predicted - y)                # error is (predicted - actual)
    squared_error = error ** 2             # we will minimize the error
    grad = [2 * error * x, 2 * error]      # using its gradient
    return grad

from mymath.linear_algebra import vector_mean

# start with random values for slope and intercept
theta = [random.uniform(-1, 1), random.uniform(-1,1)]

learning_rate = 0.001

for epoch in range(5000):
    # compute the mean of the gradients
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    # Take a step in that direction
    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert  19.9 < slope < 20.1, "slope should be about 20"
assert  4.9 < intercept < 5.1, "intercept should be 5"

# minibatch gradient descent
from typing import TypeVar, List, Iterator

T = TypeVar('T')          # this allows us to type "generic" function
                          #dataset can be a list of any single type - strs, ints, lists, 

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """Generate `batch_size`-sized minibatches from the dataset"""
    # start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset),batch_size)]

    if shuffle: random.shuffle(batch_starts)   #shuffle the butches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start: end]
        
# solve our problem again using minibatches:
theta = [random.uniform(-1, 1), random.uniform(-1,1)]

for epoch in range(1000):
    for batch in minibatches(inputs, batch_size=20):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert  19.9 < slope < 20.1, "slope should be about 20"
assert  4.9 < intercept < 5.1, "intercept should be 5"

# stochastic gradient descent, in which you take gradient steps based on training
# example at a time.

theta = [random.uniform(-1, 1), random.uniform(-1,1)]
for epoch in range(100):
    for x, y in inputs:
        grad = linear_gradient(x, y, theta)
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert  19.9 < slope < 20.1, "slope should be about 20"
assert  4.9 < intercept < 5.1, "intercept should be 5"

