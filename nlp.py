#!/usr/bin/env python
# coding: utf-8

# In[3]:


data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
("data science", 60, 70), ("analytics", 90, 3),
("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
("actionable insights", 40, 30), ("think out of the box", 45, 10),
("self-starter", 30, 50), ("customer focus", 65, 15),
("thought leadership", 35, 35)]
data


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
def text_size(total: int) -> float:
    """ equals 8 if total is 0, 28 if total is 200"""
    return 8 + total / 200 * 20

for word, job_popularity, resume_popularity in data:
    plt.text(job_popularity, resume_popularity, word, 
            ha = "center", va = "center",
            size = text_size(job_popularity + resume_popularity))
plt.xlabel("Popularity on Job Postings")
plt.ylabel("Polularity on Resumes")
plt.axis([0, 100, 0, 100])
plt.xticks([])
plt.yticks([])
plt.show()


# In[11]:


def fix_unicode(text: str) -> str:
    return text.replace(u"\u2019", "'")
                        


# In[13]:


import requests
from bs4 import BeautifulSoup
import re

def fix_unicode(text: str) -> str:
    return text.replace(u"\u2019", "'")

url = "https://www.oreilly.com/radar/what-is-data-science/"
html = requests.get(url).text
soup = BeautifulSoup(html, "html5lib")

content = soup.find("div", "article-body")   
regex = r"[\w'] + [\.]"           

document = []

for paragraph in content("p"):
    words = re.findall(regex, fix_unicode(paragraph.text))
    document.extend(words)

document


# In[ ]:




