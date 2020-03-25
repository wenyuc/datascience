from datascience.linear_algebra import Vector, dot

def step_function (x: float) -> float:
    return 1.0 if x >= 0 else 0.0

def perceptron_output(weights: Vector, bias: float, x: Vector) -> float:
    """Returns 1 if the perceptron 'fires', 0 if not"""
    calculation = dot(weights, x) + bias
    return step_function(calculation)

# an AND gate, which returns 1 if both its inputs are 1 but returns 0 if either of its inputs is 0
and_weights = [2., 2]
and_bias = -3.

assert perceptron_output(and_weights, and_bias, [1, 1] ) == 1
assert perceptron_output(and_weights, and_bias, [1, 0] ) == 0
assert perceptron_output(and_weights, and_bias, [0, 1] ) == 0
assert perceptron_output(and_weights, and_bias, [0, 0] ) == 0

# an OR gate, which returns 1 if either of its inputs is 1 but returns 0 if both of its inputs is 0
or_weights = [2., 2]
or_bias = -1.

assert perceptron_output(or_weights, or_bias, [1, 1] ) == 1
assert perceptron_output(or_weights, or_bias, [1, 0] ) == 1
assert perceptron_output(or_weights, or_bias, [0, 1] ) == 1
assert perceptron_output(or_weights, or_bias, [0, 0] ) == 0

# a NOT gate, which has one input and converts 1 to 0 and 0 to 1
not_weights = [-2.]
not_bias = 1.

assert perceptron_output(not_weights, not_bias, [0]) == 1
assert perceptron_output(not_weights, not_bias, [1]) == 0

# simple way to build a logic gate
# and_gate = min
# or_gate = max
# xor_gate = lambda x, y: 0 if x == y else 1

# sigmoid (also called logistic) is a continuous function, while step_function isn't even continuous, and
# sigmoid is a good smooth approximation of it. 
import math

def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))

def neuron_output(weights: Vector, inputs: Vector) -> float:
    # weights includes the bias term, inputs includes a 1
    return sigmoid(dot(weights, inputs))

# a neuron is simply a vector of weights whose length is one more than the number of inputs to
# that neuron (because of the bias weight). A neural network is a list (layers) of lists (neurons)
# of vectors (weights).

from typing import List

def feed_forward(neural_network: List[List[Vector]],
                 input_vector: Vector) -> List[Vector]:
    """Feed the input vector through the neural network.
       Returns the outputs of all layers (not just the last one)"""
    outputs: List[Vector] = []

    i = 0
    for layer in neural_network:
        print(i, layer)
        input_with_bias = input_vector + [1]              # add a constant
        output = [neuron_output(neuron, input_with_bias)  # compute the output
                  for neuron in layer]                    # for each neuron
        outputs.append(output)                            # add to results

        # Then the input to the next layer is the output of this one
        input_vector = output
        i += 1

    print("outputs:", outputs)
    return outputs

xor_network = [# hidden layer
               [[20., 20, -30],    # 'and' neuron
                [20., 20, -10]],   # 'or' neuron
               # output layer
               [[-60., 60, -30]]]  # '2nd' input but not 1st input' neuron

# feed_forward returns the outputs of all layers, so the [-1] gets the
# final output, and the [0] gets the value out of the resulting vector
assert 0.000 < feed_forward(xor_network, [0,0])[-1][0] < 0.001
assert 0.999 < feed_forward(xor_network, [1,0])[-1][0] < 1.000
assert 0.999 < feed_forward(xor_network, [0,1])[-1][0] < 1.000
assert 0.000 < feed_forward(xor_network, [1,1])[-1][0] < 0.001

def sqerror_gradients(network: List[List[Vector]],
                      input_vector: Vevctor,
                      target_vector: Vector) -> List[List[Vector]]:
    """Given a neural network, an input vector, and a target vector,
       make a prediction and compute the gradient of the squared error
       loss with respect to the neuron weights."""
    # forward pass
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # gradients with respect to output neuron pre-activation outputs
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]

    # gradients with respect to output neuron weights
    output_grads = [[output_deltas[i] * hidden_output
                       for hidden_output in hidden_outputs +[1]]
                    for i output_neuron in enumerate(network[-1])]

    #gradients with respect to hidden neuron pre-activation outputs
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                        dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # gradients with respect to hidden neuron weights
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                         for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]

def main():
    import random
    random.seed(0)

    # training data
    xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
    ys = [[0.], [1.], [1.], [0.]]

    # start with random weights
    network = [ # hidden layer: 2 inputs -> 2 outputs
                [[random.random() for _ in range(2 + 1)],  # 1st hidden neuron
                 [random.random() for _ in range(2 + 1)]],  # 2nd hidden neuron
                 # output layer: 2 inputs -> 1 output
                [[random.random() for _ in range(2 + 1)]]  # 1st output neuron
               ]

    from scratch.gradient_descent import gradient_step
    import tqdm

    learning_rate = 1.0

    
                 
