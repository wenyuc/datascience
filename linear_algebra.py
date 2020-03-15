from typing import List

Vector = List[float]

height_weight_age = [70,  #inches
                     170, #pounds
		     40 ]  #years

grades = [95, #exam1
          80, #exam2
	  75, #exam3
	  62 ] #exam4

def add(v: Vector, w: Vector) -> Vector:
    """ Adds corresponding elements"""
    assert len(v) == len(w), "Vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v,w)]

assert add([1,2,3],[4,5,6]) == [5,7,9]
result = add([1,2,3], [4,5,6])
print("add vectors:", result)

def subtract(v: Vector, w: Vector) -> Vector:
    """ Subtracts corresponding elements"""
    assert len(v) == len(w), "Vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v,w)]

assert subtract([5,7,9] , [1,2,3]) == [4,5,6]
result = subtract([5,7,9] , [1,2,3])
print("subtract vectors:", result)

# sum of a list of vectors
def vector_sum(vectors: List[Vector]) -> Vector:
    """ Sums of all corresponding elements"""
    # first check that vectors is not empty
    assert vectors, "no vectors provided"

    # check the vectors are all the same size
    num_elements = len(vectors[0])

    assert all(len(v) == num_elements for v in vectors), "vectors different size"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors) 
            for i in range(num_elements)]

assert vector_sum([[1,2], [3,4], [5,6], [7,8]]) == [16,20]
result = vector_sum([[1,2], [3,4], [5,6], [7,8]])
print("vectors_sum:",result)

# multiply a vector by a scalar
def scalar_multiply(c: float, v: Vector) -> Vector:
    """ Multiplies every element by c"""
    return [c * v_i for v_i in v]

result = scalar_multiply(2,[1,2,3])
print("scalar multiply:", result)
assert scalar_multiply(2, [1,2,3]) == [2,4,6]

# computes the componentwise means of a list of (same-sized) vectors
def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    
    # check vectors have same length
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "vectors different size"

    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

result = vector_mean([[1,2], [3,4], [5,6]])
print("vector mean:", result)
assert vector_mean([[1,2], [3,4], [5,6]]) == [3,4]

# computes the sum of their componentwise products
def dot(v: Vector, w: Vector) -> float:
    """computes the sum of corresponding elements products"""
    assert len(v) == len(w), "Vectors must be the same length"

    return sum(v_i * w_i for v_i, w_i in zip(v,w))

result = dot([1,2,3], [4,5,6])
print("dot:", result)
assert dot([1,2,3], [4,5,6]) == 32

# computes a vector's sum of squares
def sum_of_squares(v: Vector) -> float:
    """ Returns v_1*v_1 + v_2 * v_2 + v_3 * v_3 + ...+ v_n * v_n"""
    return dot(v,v)

result = sum_of_squares([1,2,3]) 
print("sum of square:", result)
assert sum_of_squares([1,2,3]) == 14

# computes a vector's magnitude (or length)
import math
def magnitude(v: Vector) -> float:
    """ Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))

result = magnitude([3,4])
print("magnitude:", result)
assert magnitude([3,4]) == 5

# compute the distance between two vectors
def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v,w))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between w and w"""
    return math.sqrt(squared_distance(v,w))

# another way to calculate distance between two vectors
def distance2(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v,w))

# Another type alias
Matrix =List[List[float]]

A = [[1,2,3],    # A has 2 rows and 3 columns
     [4,5,6]]

B = [[1,2],      # B has 3 rows and 2 columns
     [3,4],
     [5,6]]

from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    """ Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_columns = len(A[0]) if A else 0   # number of elements of first row
    return num_rows, num_columns

result = shape([[1,2,3], [4,5,6]])
print("shape of Matrix:", result)
assert shape([[1,2,3], [4,5,6]]) == (2,3)

def get_row(A: Matrix, i: int) -> Vector:
    """ Returns the i-th row of A (as a Vector)"""
    return A[i]   #A[i] is already the i-th row

def get_column(A: Matrix, j: int) -> Vector:
    """ Returns the j-th column of A (as a Vector)"""
    return [A_i[j]          #jth element of row A_i
            for A_i in A]   # for each row A_i

# generate a matrix
from typing import Callable

def make_matrix(num_rows: int, num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
        """Returns a num_rows x num_cols matrix
           whose (i,j)-th entry is entry_fn(i,j)"""
        return [[entry_fn(i,j)            # given i, create a list
                 for j in range(num_cols)] # [entry_fn(i,0), ...]
                 for i in range(num_rows)] # create one list for each i

# create an identity matrix

def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n,n, lambda i,j: 1 if i==j else 0)

result = identity_matrix(5)
print("identity matrix:")
print(result)
assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
			      [0, 0, 0, 0, 1]]
data = [[70, 170, 40],
        [65, 120, 26],
        [77, 250, 19],
        # ....
       ]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
              (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#            user 0  1  2  3  4  5  6  7  8  9
#
friend_matrix = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # user 0
                 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # user 1
                 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # user 2
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # user 3
                 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # user 4
                 [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # user 5
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 6
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 7
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # user 8
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]  # user 9

assert friend_matrix[0][2] == 1, "0 and 2 are friends"
assert friend_matrix[0][8] == 0, "0 and 8 are not friends"
                 