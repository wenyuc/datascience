#!/usr/local/bin/python3
import numpy as np

# boolean indexing
names = np.array(['Bob', 'Joe', 'will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.rand(7,4)

print(names)
print(data)
print("\n")
print("names == Bob")
print(names == 'Bob')
print("data [names == Bob]")
print(data[names == 'Bob'])

print("names = will and data[will]")
print(names == 'will')
print(data[names == 'will'])

print("data[names ==bob], 2: and 3")
print(data[names == 'Bob', 2:])
print(data[names == 'Bob', 3])

print("\ndata[~Bob]:")
print(data[~(names == 'Bob')])

print("\nuse cond")
cond = names == 'Bob'
print(data[~cond])

print("\nmask == Bob or will")
mask = (names == 'Bob') | (names == 'will')
print(mask)
print(data[mask])
print("\n")
print(data)

print("larger than 0 to 0")
data[data > 0] = 0
print(data)

print("not Joe to 7")
data[names != 'Joe'] =7
print(data)

print("8 x 4 array with empty")
arr = np.empty((8,4))
print(arr)

print("\n change to i")
for i in range(8):
    arr[i] = i
print(arr)

print("\nselect rows")
print(arr[[4,3,0,7]])
print(arr[[-3, -5, -7]])

print("\nreshape")
arr = np.arange(32).reshape(8,4)
print(arr)
print(arr[[1,5,7,2],[0,3,1,2]])

print("\ntranspose")
arr = np.random.randn(6,3)
print(arr)
print("arr.T dot arr")
print(np.dot(arr.T, arr))

print("transpose with tuple of axis  number")
arr = np.arange(24).reshape((2,3,4))
print(arr)
print("\nthe axes reordered with 2nd axis first, ")
print("1st axis second, the last axis unchanged.")
print(arr.transpose(1,0,2))

print("\nswapaxes")
arr = np.arange(24).reshape(2,3,4)
print(arr)
print("swapaxes(1,2)")
print(arr.swapaxes(1,2))
print("\nswapaxes(0,1)")
print(arr.swapaxes(0,1))

