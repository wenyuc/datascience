from typing import List
from collections import Counter

def tokenize(document: str) -> List[str]:
    """Split on whitespace"""
    return document.split()

def word_count_old(documents: List[str]):
    """Word count not useing MapReduce"""
    return Counter(word 
                    for document in documents 
                    for word in tokenize(document))

##########  mapper and reducer utility##############
from typing import Iterator, Iterable, Tuple

def wc_mapper(document: str) -> Iterator[Tuple[str, int]]:
    """For each word in the document, emit (word, 1)"""
    for word in tokenize(document):
        yield (word, 1)

def wc_reducer(word: str, counts: Iterable[int]) -> Iterator[Tuple[str, int]]:
    """Sum up the counts for a word"""
    yield (word, sum(counts))

# simple MapReduce using utilities

from collections import defaultdict

def wc_count(documents: List[str]) -> List[Tuple[str, int]]:
    """Count the words in the input documents using MapReduce"""

    collector = defaultdict(list)    # to store grouped values

    for document in documents:
        for word, count in wc_mapper(document):
            collector[word].append(count)
    
    # print(collector.items())

    return [output 
            for word, counts in collector.items()
            for output in wc_reducer(word, counts)]

################### Generic Mapper, Reducer ###############
from typing import Callable, Iterable, Any, Tuple

# A key/value pair is just a 2-tuple
KV = Tuple[Any, Any]

# A Mapper is a function that returns an Iterable of key/value
Mapper = Callable[..., Iterable[KV]]

# A Reducer is a fuction that takes a key and an iterable of values
# and returns a key/value pair
Reducer = Callable[[Any, Iterable], KV]

# abstract agreggation for reducer
def values_reducer(values_fn: Callable) -> Reducer:
    """Returns a reducer that just applies values_fn to its values"""
    def reduce(key, values: Iterable) -> KV:
        return(key, values_fn(values))

    return reduce

# Now, we can have many kinds of reducer
sum_reducer = values_reducer(sum)
max_reducer = values_reducer(max)
min_reducer = values_reducer(min)
distinct_reducer = values_reducer(lambda values: set(values))
count_distinct_reducer = values_reducer(lambda values: len(set(values)))

# assert
assert sum_reducer("key", [1,2,3,4]) == ("key", 10)
assert min_reducer("key", [1,2,3,4]) == ("key", 1)
assert max_reducer("key", [1,2,3,4]) == ("key", 4)
assert distinct_reducer("key", [1,2,1,3,2,4,3,1,5]) == ("key", {1,2,3,4,5})
assert count_distinct_reducer("key", [1,2,1,3,2,4,3,1,5]) == ("key", 5)

# a general map_reduce function
def map_reduce(inputs: Iterable,
                mapper: Mapper,
                reducer: Reducer) -> List[KV]:
    """Runs MapReduce on the inputs using mapper and reducer"""
    collector = defaultdict(list)

    for input in inputs:
        for key, value in mapper(input):
            collector[key].append(value)
    
    print(collector.items())

    return [output
            for key, values in collector.items()
            for output in reducer(key, values)]

def main():
    documents = ["data science", "big data", "science fiction"]
    
    # use the old way to word count
    # word_count = word_count_old(line)

    # use simple wc_mapper and wc_reducer
    # word_count = wc_count(documents)

    # use the general MapReduce with wc_reducer
    #word_count = map_reduce(documents, wc_mapper, wc_reducer)

    # use the general MapReduce with sum_reducer
    #word_max = map_reduce(documents, wc_mapper, sum_reducer)
    #print(word_max)

    import datetime

    status_updates = [
        {"id": 2,
         "username" : "joelgrus",
         "text" : "Should I write a second edition of my data science book?",
         "created_at" : datetime.datetime(2018, 2, 21, 11, 47, 0),
         "liked_by" : ["data_guy", "data_gal", "mike"] },
        {"id": 3,
         "username" : "wenyuc",
         "text" : "continue study data science and big data.",
         "created_at" : datetime.datetime(2020, 4, 19, 11, 47, 0),
         "liked_by" : ["reducemap", "python", "mysql", "neural network", "reducemap"] },
         # ...
    ]

    # 1. count how many data science updates there are on each day of the week
    def data_science_day_mapper(status_update: dict) -> Iterable:
        """Yields (day_of_week, 1) if status_update contains "data science" """

        if "data science" in status_update["text"].lower():
            day_of_week = status_update["created_at"].weekday()
            yield (day_of_week, 1)

    data_science_days = map_reduce(status_updates, 
                                    data_science_day_mapper, 
                                    sum_reducer)
    print(data_science_days)

    # 2. to find out for each user the most common word that she puts in her status update.
    def words_per_user_mapper(status_update: dict) -> Iterable:
        """ Yields(username, (word,1) """
        user = status_update["username"]

        for word in tokenize(status_update["text"]):
            word = word.rstrip('?').rstrip('.')
            yield(user, (word, 1))
    
    def most_popular_word_reducer(user: str, words_and_counts: Iterable[KV]):
        """ Given a sequence of (word, count) pairs,
            returns the word with the highest total count. """
        word_counts = Counter()
        for word, count in words_and_counts:
            word_counts[word] += count
          
        word, count = word_counts.most_common(1)[0]
        
        yield (user, (word, count))
        
    user_words = map_reduce(status_updates, words_per_user_mapper, most_popular_word_reducer)
    
    print(user_words)
    
    # 3. liker mapper
    def liker_mapper(status_update: dict):
        user = status_update["username"]
        for liker in status_update["liked_by"]:
            yield (user, liker)
    
    distinct_likers_per_user = map_reduce(status_updates, liker_mapper, count_distinct_reducer)
    
    print(distinct_likers_per_user)
    
    # 4. Matrix manipulation
    from typing import NamedTuple
    
    class Entry(NamedTuple):
        name: str
        i: int
        j: int
        value: float
    
    def matrix_multiply_mapper(num_rows_a: int, num_cols_b: int) -> Mapper:
        """ C[x][y] = A[x][0] * B[0][y] + A[x][1] * B[1][y] + ... A[x][m] * B[m][y]
            So, an element A[i][j] goes into every C[i][y] with coef B[j][y]
            and an element B[i][j] goes into every C[x][j] with coef A[x][i]"""
        
        def mapper(entry: Entry):
            if entry.name == "A":
                for y in range(num_cols_b):
                    key = (entry.i, y)              # which element of C
                    value = (entry.j, entry.value)  # which entry in the sum
                    yield (key, value)
            else:
                for x in range(num_rows_a):
                    key = (x, entry.j)              # which element of C
                    value = (entry.i, entry.value)  # which entry in the sum
                    yield (key, value)
     
    def matrix_multiply_reducer(key: Tuple[int, int],
                                indexed_values: Iterable[Tuple[int, int]]):
        results_by_index = defaultdict(list)
        
        for index, value in indexed_values:
            result_by_index[index].append(value)
        
        # multiply the values for positions with two values
        # (one from A, and one from B) and sum them up
        
        sumproduct = sum(value[0] * value[1]
                         for values in result_by_index.values()
                         if len(values) == 2)
        
        if sumproduct != 0.0:
            yield (key, sumproduct)
            
    """ A = [[3, 2, 0],
           [0, 0, 0]]
       
        B = [[4, -1, 0],
             [10, 0, 0],
             [0, 0, 0]] """

    entries = [Entry("A", 0, 0, 3), Entry("A", 0, 1, 2), Entry("B",0, 0, 4),
               Entry("B", 0, 1, -1), Entry("B", 1, 0, 10)]

    mapper = matrix_multiply_mapper(num_rows_a = 2, num_cols_b = 3)

    reducer = matrix_multiply_reducer
 
    assert (set(map_reduce(entries, mapper, reducer)) == 
            {((0, 1), -3), ((0, 0), 32)})
 
if __name__ == "__main__": main()
