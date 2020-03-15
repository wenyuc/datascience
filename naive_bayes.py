# scikit-learn contains a BernoulliNB model that implements the same
# Naive Bayes algorithm implemented here, as well as other varitions
# on the model.
from typing import Set
import re

def tokenize(text: str) -> Set[str]:
    text = text.lower()                         # convert to lowercase,
    all_words = re.findall("[a-z0-9']+", text)  # extract the words, and
    return set(all_words)                       # remove duplicates

assert tokenize("Data Science is science") == {"data", "science", "is"}

from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool

from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k     #smoothing factor

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int]= defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # increment message counts
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probability(self, token: str) -> Tuple[float, float]:
        """ returns P(token|spam) and P(token|ham)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_token = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probability(token)

            # if *token* appears in the message,
            # add the log probability of seeing it
            if token in text_token:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)

        return prob_if_spam / (prob_if_spam + prob_if_ham)

messages = [Message("spam rules", is_spam = True),
            Message("ham rules", is_spam = False),
            Message("hello ham", is_spam = False)]

model = NaiveBayesClassifier(k = 0.5)
model.train(messages)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

print(model.tokens)
print(model.spam_messages)
print(model.ham_messages)
print(model.token_spam_counts)
print(model.token_ham_counts)

text = "hello spam"

prob_if_spam = [
    (0.5 + 1) / (2 * 0.5 + 1),      # "spam" present
    1 - (0.5 + 0) / (2 * 0.5 + 1),  # "ham" not present
    1 - (0.5 + 1) / (2 * 0.5 + 1),  # "rules" not present
    (0.5 + 0) / (2 * 0.5 +1)]       # "hello" present

prob_if_ham = [
    (0.5 + 0) / (2 * 0.5 + 2),      # "spam" present
    1 - (0.5 + 2) / (2 * 0.5 + 2),  # " ham" not present
    1 - (0.5 + 1) / (2 * 0.5 + 2),  # "rules" not present
    (0.5 + 1) / (2 * 0.5 + 2)]      # "hello" prsent

p_if_spam = math.exp(sum(math.log(p) for p in prob_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in prob_if_ham))

# should be 0.83
#assert model.predict(text) == p_if_spam / (p_if_spam + p_if_ham)
print(model.predict(text))

# get and extract tar file
from io import BytesIO     # treat bytes as a file
import requests            # to download the files
import tarfile             # in tar.bz2 format

if 0:
    BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
    FILES = ["20021010_easy_ham.tar.bz2",
             "20021010_hard_ham.tar.bz2",
             "20021010_spam.tar.bz2"]

    # where the data will end up
    OUTPUT_DIR = 'spam_data'

    for filename in FILES:
        # use requests to get the file contents at each URL
        content = requests.get(f"{BASE_URL}/{filename}").content

        # wrap the in-memory bytes so we can use them as a "file"
        fin = BytesIO(content)

        # and extract all the files to the specified output dir
        with tarfile.open(fileobj = fin, mode='r:bz2') as tf:
            tf.extractall(OUTPUT_DIR)

import glob, re

path = "spam_data/*/*"

data: List[Message] = []

# glob.glob returns every filename that matches the wildcarded path
for filename in glob.glob(path):
    is_spam = "ham" not in filename

    # there are some garbage characters in the emails, the errors = 'ignore'
    # skips them instead of raising an exception.
    with open(filename, errors = 'ignore') as email_file:
        for line in email_file:
            if line.startswith("Subject: "):
                subject = line.lstrip("Subject ")
                data.append(Message(subject, is_spam))
                break     # done with this file

# split the data into training data and test data
import random
from mymath.machine_learning import split_data

random.seed(0)
train_messages, test_messages = split_data(data, 0.75)

model = NaiveBayesClassifier()
model.train(train_messages)

from collections import Counter
predictions = [(message, model.predict(message.text))
               for message in test_messages]

# Assume that spam_probability > 0.5 corresponds to spam prediction
# and count the combinations of (actual is_spam, predicted is_spam)
confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                           for message, spam_probability in predictions)
print(confusion_matrix)

def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    # we probably shouldn't call private methods, but it's for good couae
    prob_if_spam, prob_if_ham = model._probability(token)

    return prob_if_spam / (prob_if_spam + prob_if_ham)

words = sorted(model.tokens, key = lambda t: p_spam_given_token(t, model))

print("spammiest_words:", words[-10:])
print("hammiest_words:", words[:10])
def main():
    text = "message to lowercase, then c000 use re.findall to extract “words” consisting of letters"
    words = tokenize(text)
    print(words)
    
if __name__ == "__main__": main()
