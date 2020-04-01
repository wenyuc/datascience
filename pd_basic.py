#!/usr/local/bin/python3
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

obj = pd.Series([4,7, -5, 3])
print(obj)
print("values and index")
print(obj.values)
print(obj.index)
obj2 = pd.Series([4,7, -5, 3], index = ['d', 'b', 'c', 'a'])
print(obj2)
print("values and index")
print(obj2.values)
print(obj2.index)
# functions, such as filtering, scalar multiplication, etc
print(obj2[obj2>0])
print(obj2 * 2)
print(np.exp(obj2))

# use like a dict
print('b' in obj2)
print('e' in obj2)

# create a Series by passing a dict
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 18000}
obj3 = pd.Series(sdata)
print(obj3)

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index = states)
print(obj4)
print(pd.isnull(obj4))
print(pd.notnull(obj4))
print(obj4.isnull())

# arithmetic operations
print(obj3 + obj4)

# Series object name and its index name
obj4.name = "population"
obj4.index.name = "state"
print(obj4)

# alter a Series's index 
obj.index = ['Bob', "Steve", 'Jeff', "Ryan"]
print(obj)

# index objects are immutable
obj = pd.Series(range(3), index = ['a', 'b', 'c'])
index = obj.index
print(index)
print(index[1:])

labels = pd.Index(np.arange(3))
print(labels)
obj5 = pd.Series([1.5, -2.5, 0], index = labels)
print(obj5)
print(obj5.index is labels)

# reindex
obj6 = pd.Series([4.5, 7.2, -5.3, 3.6], index = ['d', 'b', 'c', 'a'])
print(obj6)
obj6 = obj6.reindex(['a', 'b', 'c', 'd', 'e'])
print(obj6)

# DataFrame
# outer dict keys as the columns
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2003, 2004, 2005],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
print(frame)

# the head method returns only the first 5 rows
print(frame.head())

# change the sequence of columns
frame2 = pd.DataFrame(data, columns=['year', 'pop', 'state'])
print(frame2)

# if a column isn't contained in the dict, it will appear with missing values

frame3 = pd.DataFrame(data, columns = ['year', 'pop', 'state', 'debt'],
                            index = ['one', 'two', 'three', 'four',
                                     'five', 'six'])
print(frame3)
print(frame3.columns)

# retrieve a column
print("retrieve a column")
print(frame3['state'])
print(frame3.year)

# retrieve a row
print("retrieve a row")
print(frame3.loc['three'])

frame3['debt'] = 16.5
print(frame3)
frame3['debt'] = np.arange(6.)
print(frame3)

# assign a Series, 
val = pd.Series([-1.2, -1.5, -1.7], index = ['two', 'four', 'five'])
frame3['debt'] = val
print(frame3)

# add a new column
frame3['eastern'] = frame3.state == 'Ohio'
print(frame3)

# remove a column
del frame3['eastern']
print(frame3.columns)

# nested dict of dicts
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame4 = pd.DataFrame(pop, index=[2000, 2001, 2002])
print(frame4)

# transpose
print("transpose")
print(frame4.T)

print(frame4)
pdata = {'Ohio': frame4['Ohio'][:-1],
         'Nevada': frame4['Nevada'][:-1]}
print(pd.DataFrame(pdata))
