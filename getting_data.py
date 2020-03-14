# getting_data.py

# Just stick some data there
#with open('email_addresses.txt', 'w') as f:    #the file auto closed at the end of with block
#    f.write("chenwenyu027@gmail.com\n")
#    f.write("chenwenyu077@live.com\n")
#    f.write("wenyu.chen@nokia-sbell.com\n")
    
def get_domain(email_address: str) -> str:
    """Split on '@' and return the last piece"""
    return email_address.lower().split("@")[-1]

print(get_domain("chenwenyu077@live.com"))
print(get_domain("wenyu.chen@nokia-sbell.com"))

#with open('tab_delimited_stock_prices.txt', 'w') as f:
#    f.write("""6/20/2014\tAAPL\t90.91
#6/20/2014\tMSFT\t41.68
#6/20/2014\tFB\t64.5
#6/19/2014\tAAPL\t91.86
#6/19/2014\tMSFT\t41.51
#6/19/2014\tFB\t64.34
#""")

import csv

def process(date: str, symbol: str, closing_price: float) -> None:
    # Imaginge that this function actually does something.
    #assert closing_price > 0.0
    print(date, symbol, closing_price)
    
with open('tab_delimited_stock_prices.txt', 'r') as f:
    tab_reader = csv.reader(f, delimiter='\t')
    for row in tab_reader:
        date = row[0]
        symbol = row[1]
        closing_price = row[2]
        process(date, symbol, closing_price)

#with open('colon_delimited_stock_prices.txt', 'w') as f:
#    f.write("""date:symbol:closing_price
#6/20/2014:AAPL:90.91
#6/20/2014:MSFT:41.68
#6/20/2014:FB:64.5
#""")

with open('colon_delimited_stock_prices.txt','r') as f:
    colon_reader = csv.DictReader(f, delimiter = ':')
    for dict_row in colon_reader:
        date = dict_row["date"]
        symbol = dict_row["symbol"]
        closing_price = float(dict_row["closing_price"])
        process(date, symbol, closing_price)
        
today_prices = {'AAPL': 90.91, 'MSFT': 41.68, 'FB': 64.5}

with open('comma_delimited_stock_prices.txt', 'w') as f:
    csv_writer = csv.writer(f, delimiter = ',')
    for stock, price in today_prices.items():
        csv_writer.writerow([stock, price])

from bs4 import BeautifulSoup
import requests

# I put the relevant HTML file on GitHub. In order to fit
# the URL in the book I had to split it across two lines.
url = "https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html"
html = requests.get(url).text
soup = BeautifulSoup(html,'html5lib')

# to find the fist <p> tag and its content
first_paragraph = soup.find('p')    # or just soup.p
first_paragraph_text = soup.p.text
first_paragraph_words = soup.p.text.split()
print(first_paragraph, first_paragraph_text, first_paragraph_words)

first_paragraph_id = soup.p['id']
first_paragraph_id2 = soup.p.get('id')
print(first_paragraph_id,  first_paragraph_id2)

# to get multiple tags at once
all_paragraphs = soup.find_all('p')  # or just soup('p')
paragraphs_with_ids = [p for p in soup('p') if p.get('id')]
print(all_paragraphs, paragraphs_with_ids)

# to find tags with a specific class
important_paragraphs = soup('p',{'class': 'important'})
important_paragraphs2 = soup('p','important')
important_paragraphs3 = [p for p in soup('p')
                         if 'important' in p.get('class',[])]

print(important_paragraphs, important_paragraphs2, important_paragraphs3)

# to get spans inside divs
spans_inside_divs = [span
                     for div in soup('div')
                     for span in div('span')]
print(spans_inside_divs)

# to collect all of the URLs linked to
from bs4 import BeautifulSoup
import requests

url = "https://www.house.gov/representatives"
text = requests.get(url).text
soup = BeautifulSoup(text, 'html5lib')

all_urls = [a['href']
            for a in soup('a')
            if a.has_attr('href')]

print(all_urls)
print(len(all_urls))


import re
# must start with http:// or https://
# must end with .house.gov or .house.gov/
regex = r"^https?://.*\.house\.gov/?$"

#Let's write some tests!
assert re.match(regex, "http://joel.house.gov")
assert re.match(regex, "https://joel.house.gov")
assert re.match(regex, "http://joel.house.gov/")
assert re.match(regex, "https://joel.house.gov/")
assert not re.match(regex, "joel.house.gov")
assert not re.match(regex, "http://joel.house.com")
assert not re.match(regex, "https://joel.house.gov/biography")

#And now apply
good_urls = [url for url in all_urls if re.match(regex, url)]
print(good_urls)
print(len(good_urls))

# to get rid of duplicate ones
good_urls = list(set(good_urls))
print(good_urls)
print(len(good_urls))

html = requests.get('https://jayapal.house.gov').text
soup = BeautifulSoup(html, 'html5lib')

# use a set because the links might appear multiple times.
links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}
print(links)

if 0 :
    from typing import Dict, Set

    press_releases: Dict[str, Set[str]] = {}

    for house_url in good_urls:
        html = requests.get(house_url).text
        soup = BeautifulSoup(html, 'html5lib')

        pr_links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}
        print(f"{house_url}: {pr_links}")

        press_releases[house_url] = pr_links

    print(press_releases)

def paragraph_mentions(text: str, keyword: str) -> bool:
    """ Returns True if <p> inside the text mentions {keyword}"""
    soup = BeautifulSoup(text, "html5lib")
    paragraphs = [p.get_text() for p in soup('p')]

    return any(keyword.lower() in paragraph.lower()
               for paragraph in paragraphs)

text = """<body><h1>Facebook</h1><p>Twitter</p>"""
#result = paragraph_mentions(text, "twitter")
#print("twitter in <p>", result)
#result = paragraph_mentions(text, "facebook")
#print("facebook in <p>", result)
assert paragraph_mentions(text, "twitter")      # is inside <p>
assert not paragraph_mentions(text, "facebook") # not inside <p>

# get "data" in press_releases
if 0 :
    for house_url, pr_link in press_releases.items():
        for pr_link in pr_links:
            url = f"{house_url}/{pr_link}"
            text = requests.get(url).text

            if paragraph_mentions(text, "data"):
                print(f"{house_url}")
                break    #done with this house_url

# serialization
import json
serialized = """{"title": "Data Science Book",
                 "author": "Joel Grus",
                 "publicationYear": 2019,
                 "topics": [ "data", "science", "data science" ] }"""

#parse the JSON to create a Python dict
deserialized = json.loads(serialized)
print(deserialized)
assert deserialized["publicationYear"] == 2019
assert "data science" in deserialized["topics"]

github_user = "joelgrus"
endpoint = f"https://api.github.com/users/{github_user}/repos"
print(endpoint)
repos = json.loads(requests.get(endpoint).text)
print(repos)

from collections import Counter
from dateutil.parser import parse

dates = [parse(repo["created_at"]) for repo in repos]
print("created at", dates)
month_counts = Counter(date.month for date in dates)
print("month_counts:", month_counts)
weekday_counts = Counter(date.weekday() for date in dates)
print("weekday_counts:", weekday_counts)

# get the language of my last five repos
last_5_repos = sorted(repos,
                      key = lambda r: r["pushed_at"],
                      reverse = True)[:5]
last_5_languages = [repo["language"] for repo in last_5_repos]
print(last_5_languages)

#list of python api wrappers:
# https://github.com/realpython/list-of-python-api-wrappers


