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
    """Run MapReduce on the inputs using mapper and reducer"""
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
    word_max = map_reduce(documents, wc_mapper, sum_reducer)
    print(word_max)

    import datetime

    status_updates = [
        {"id": 2,
         "username" : "joelgrus",
         "text" : "Should I write a second edition of my data science book?",
         "created_at" : datetime.datetime(2018, 2, 21, 11, 47, 0),
         "liked_by" : ["data_guy", "data_gal", "mike"] },
         # ...
    ]

    # count how many data science updates there are on each day of the week
    def data_science_day_mapper(status_update: dict) -> Iterable:
        """Yields (day_of_week, 1) if status_update contains "data science" """

        if "data science" in status_update["text"].lower():
            day_of_week = status_update["created_at"].weekday()
            yield (day_of_week, 1)

    data_science_days = map_reduce(status_updates, 
                                    data_science_day_mapper, 
                                    sum_reducer)
    print(data_science_days)

    # to find out for each user the most common word that she puts in her status update.
    def words_per_user_mapper(status_update: dict) -> Iterable:
        """ Yields(username, (word,1) """
        user = status_update["username"]

        for word in tokenize(status_update["text"]):
            yield(user, (word, 1))

if __name__ == "__main__": main()
