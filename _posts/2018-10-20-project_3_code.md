
# Using Reddit's API for Predicting Comments

In this project, we will practice two major skills. Collecting data via an API request and then building a binary predictor.

As we discussed in week 2, and earlier today, there are two components to starting a data science problem: the problem statement, and acquiring the data.

For this article, your problem statement will be: _What characteristics of a post on Reddit contribute most to what subreddit it belongs to?_

Your method for acquiring the data will be scraping threads from at least two subreddits. 

Once you've got the data, you will build a classification model that, using Natural Language Processing and any other relevant features, predicts which subreddit a given post belongs to.

### Scraping Thread Info from Reddit.com

#### Set up a request (using requests) to the URL below. 

*NOTE*: Reddit will throw a [429 error](https://httpstatuses.com/429) when using the following code:
```python
res = requests.get(URL)
```

This is because Reddit has throttled python's default user agent. You'll need to set a custom `User-agent` to get your request to work.
```python
res = requests.get(URL, headers={'User-agent': 'YOUR NAME Bot 0.1'})
```


```python
# Imports - used or otherwise.
import pandas as pd
import requests
import json
import time
import regex as re
import praw
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve
```


```python
# Create the URL variables
URL_ds = "http://www.reddit.com/r/datascience.json"
URL_stats = "https://www.reddit.com/r/statistics.json"
```


```python
# Authenticating via OAuth for praw
reddit = praw.Reddit(client_id='AOJTLQLavhOXPg',
                     client_secret='eS08QOpy2lWh37qkVBGlN7yMjRI',
                     username='TCRAY_DSI',
                     password='dsi123',
                     user_agent='TK Bot 0.1')
```


```python
# Check
print(reddit.user.me())
```

    TCRAY_DSI



```python
# Create subs for praw:
sub_ds = reddit.subreddit('datascience')
sub_stats = reddit.subreddit('statistics')
```


```python
# Create top pulls
top_ds = sub_ds.top(time_filter='year')
top_stats = sub_stats.top(time_filter='year')
```


```python
# These were used and attempted (to success) before creating the loop 
# Request the JSON files
# I did them in seperate cells to space out the scrapping, so reddit wouldn't throw a 429 error
# res_ds = res.get(URL_ds, headers={'User-agent': 'TK Bot 0.1'})
```


```python
# res_stats = requests.get(URL_stats, headers={'User-agent': 'TK Bot 0.1'})
```


```python
# res_stats.status_code
```

#### Use `res.json()` to convert the response into a dictionary format and set this to a variable. 

```python
data = res.json()
```


```python
# These were used and attempted (to success) before creating the loop 
# Convert the JSON responses
# data_ds = res_ds.json()
# data_stats = res_stats.json()
```


```python
# Check out data
# data_ds
# data_stats
```


```python
# Testing adding nested dictionaries to each other
# doubling_up = data_ds['data']['children'] + data_ds['data']['children'] 
# doubling_up
```

#### Getting more results

By default, Reddit will give you the top 25 posts:

```python
print(len(data['data']['children']))
```

If you want more, you'll need to do two things:
1. Get the name of the last post: `data['data']['after']`
2. Use that name to hit the following url: `http://www.reddit.com/r/boardgames.json?after=THE_AFTER_FROM_STEP_1`
3. Create a loop to repeat steps 1 and 2 until you have a sufficient number of posts. 

*NOTE*: Reddit will limit the number of requests per second you're allowed to make. When you create your loop, be sure to add the following after each iteration.

```python
time.sleep(3) # sleeps 3 seconds before continuing```

This will throttle your loop and keep you within Reddit's guidelines. You'll need to import the `time` library for this to work!


```python
# Check out length
# print(len(data_ds['data']['children']))
# print(len(data_stats['data']['children']))
```


```python
# Test the last post pull
# data_ds['data']['after']
```


```python
# For DS set - previously ran to generate CSV
url_ds = "https://www.reddit.com/r/datascience.json"
data_ds = []
total = []
next_get = ''

# I went with 40 b/c 40 * 25 = 1000 posts total
for i in range(40):

    # Request get
    res = requests.get(url_ds+next_get, headers={'User-agent': 'TK Bot 0.1'})
    
    # Convert the JSON
    new_dict = res.json()
    
    # Add to already collected data set
    data_ds.extend(new_dict['data']['children'])
    
    # Collect 'after' from new dict to generate next URL
    new_url_end = str(new_dict['data']['after'])
    
    # Generate the next URL
    next_get = '?after='+new_url_end
    
    # CSV add/update along with DF creation
    # Chose greater than 0 so the else executes on the first iteration
    if i > 0:
        # Read in previous csv for comparision/add
        # Establish current DF - left over from previous way of running
        # past_posts = pd.read_csv('data_ds.csv')
        # current_df = pd.DataFrame(data_ds)
        
        # Append new and old
        total = pd.DataFrame(data_ds)
        
        # Convert to DF and save to new csv file
        pd.DataFrame(total).to_csv('data_ds.csv', index = False)
    
    else:
        pd.DataFrame(data_ds).to_csv('data_ds.csv', index = False)
        
    # Sleep to fit within Reddit's pull limit
    time.sleep(3)
```


```python
# For stats set - previously ran to generate CSV
url_stats = "https://www.reddit.com/r/statistics.json"
data_stats = []
total = []
next_get = ''

# I went with 40 b/c 40 * 25 = 1000 posts total
for i in range(40):

    # Request get
    res = requests.get(url_stats+next_get, headers={'User-agent': 'TK Bot 0.1'})
    
    # Convert the JSON
    new_dict = res.json()
    
    # Add to already collected data set
    data_stats.extend(new_dict['data']['children'])
    
    # Collect 'after' from new dict to generate next URL
    new_url_end = str(new_dict['data']['after'])
    
    # Generate the next URL
    next_get = '?after='+new_url_end
    
    # CSV add/update along with DF creation
    # Chose greater than 0 so the else executes on the first iteration
    if i > 0:
        # Read in previous csv for comparision/add
        # Establish current DF - left over from previous way of running
        # past_posts = pd.read_csv('data_stats.csv')
        # current_df = pd.DataFrame(data_stats)
        
        # Append new and old
        total = pd.DataFrame(data_stats)
        
        # Convert to DF and save to new csv file
        pd.DataFrame(total).to_csv('data_stats.csv', index = False)
    
    else:
        pd.DataFrame(data_stats).to_csv('data_stats.csv', index = False)
        
    # Sleep to fit within Reddit's pull limit
    time.sleep(3)
```


```python
# This was an older attempt at writing the function, that I scrapped and decided to start fresh on:
# url_ds = "https://www.reddit.com/r/datascience.json?after=" + last_post_ds

# for i in range(25):
#     # Get the name of the last post
#     last_post_ds = data_ds['data']['after']
    
#     # Set the url from the last post
#     new_url_ds = "https://www.reddit.com/r/datascience.json?after=" + last_post_ds
    
#     # Perform request get
#     new_res_ds = res.get(new_url_ds, headers={'User-agent': 'TK Bot 0.1'})

#     # Convert the JSON to a dict
#     new_data_ds = new_res_ds.json()

#     # Add the new dict to the already existing one
#     data_ds.update(new_data_ds)
#     data_ds['data']['children'] = data_ds['data']['children'] + new_data_ds['data']['children']
#     data_ds['data']['after'] = new_data_ds['data']['after']
    
#     # Sleep
#     # time.sleep(3)
```


```python
# Next few cells devoted to understanding how to generate a combined dict
# new_data_ds.items()
```


```python
# OG_ds_data = data_ds.copy()
```


```python
# new_data_ds = new_res_ds.json()
# data_ds.update(new_data_ds)
```


```python
# Testing adding nested dictionaries to each other
# doubling_up = data_ds['data']['children'] + data_ds['data']['children'] 
# doubling_up
```

### Save your results as a CSV
You may do this regularly while scraping data as well, so that if your scraper stops of your computer crashes, you don't lose all your data.


```python
# My loop in the previous cell completes this step.
```

### Read my files back in and clean them up / EDA



```python
%pwd
```




    '/Users/tomkelly/Desktop/general_assembly/DSI-US-5/project-3'




```python
df_ds = pd.read_csv('./data_ds.csv')
df_stats = pd.read_csv('./data_stats.csv')
```


```python
# 983 DS posts vs 978 stats posts
# df_ds.shape[0]
df_stats.shape[0]
```




    978




```python
# for i in df_ds.shape[0]
# Testing what I want to loop
df_ds['body'] = pd.Series(re.findall('(?<=selftext).{4}(.*).{4}(?=author_fullname)', df_ds['data'][0]))
```


```python
df_ds.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>kind</th>
      <th>body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>The Mod Team has decided that it would be nice...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
type(df_ds['body'].iloc[0,])
```




    str




```python
# To pull out the body of the post and make it a new column
for i in range(0, df_ds.shape[0]):
    try: #Since regex makes it a list, this helps deal with nulls
        df_ds['body'][i] = re.findall('(?<=selftext).{4}(.*).{4}(?=author_fullname)', df_ds['data'][i])[0]
    except:
        df_ds['body'][i] = ''
```


```python
df_ds.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>kind</th>
      <th>body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>The Mod Team has decided that it would be nice...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>\n\nWelcome to this week's 'Entering &amp;amp; Tr...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>I'm working on making a list of Machine Learni...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# For some reason, wrapping it in pd.Series makes this work before I loop it
try:
    df_ds['title'] = pd.Series(re.findall('(?<= .title).{4}(.*).{4}(?=link_flair_richtext)', df_ds['data'][0]))[0]
except:
    df_ds['title'] = ''
```


```python
# To pull out the title of the post and make it a new column
for i in range(0, df_ds.shape[0]):
    try:
        df_ds['title'][i] = re.findall('(?<= .title).{4}(.*).{4}(?=link_flair_richtext)', df_ds['data'][i])[0]
    except:
        df_ds['title'][i] = ''
```


```python
df_stats.shape[0]
```




    978




```python
df_ds.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>kind</th>
      <th>body</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>The Mod Team has decided that it would be nice...</td>
      <td>DS Book Suggestions/Recommendations Megathread</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>\n\nWelcome to this week's 'Entering &amp;amp; Tr...</td>
      <td>Weekly 'Entering &amp;amp; Transitioning' Thread. ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td></td>
      <td>Mo Data, Mo Problems. Everyone always talks ab...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td></td>
      <td>Make “Fairness by Design” Part of Machine Lear...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>I'm working on making a list of Machine Learni...</td>
      <td>Papers with Code</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Looks like the body/title got pulled in as a list, turning it into a str
# This is leftover from an older method
# for w in range(0,df_ds['body'].shape[0]):
#     df_ds['body'][w] = str(df_ds['body'][w])
```


```python
# Additional Clean-up - DS
df_ds['body'] = df_ds['body'].map(lambda x: x.replace('\\n',''))
df_ds['body'] = df_ds['body'].map(lambda x: x.replace('\n',''))
df_ds['body'] = df_ds['body'].map(lambda x: x.replace('\\',''))
df_ds['body'] = df_ds['body'].map(lambda x: x.replace("\\'","'"))
# df_ds['body'] = [w.replace('/n', '') for w in df_ds['body']]
```


```python
df_ds['body']
```




    0      The Mod Team has decided that it would be nice...
    1       Welcome to this week's 'Entering &amp; Transi...
    2                                                       
    3                                                       
    4      I'm working on making a list of Machine Learni...
    5      I do most of my work in Python. Building the m...
    6      [Project Link](https://github.com/HiteshGorana...
    7      Before I got hired, my company had a contracto...
    8      I'm looking for an open-source web-based tool ...
    9      I've been reading around online a bit as to wh...
    10     I am new to time series data, so bear with me....
    11                                                      
    12     Hey all, Do people have recommendations for pi...
    13     I am quite old (23), but would like to become ...
    14                                                      
    15     I know that python and R are the standard lang...
    16     Which tools and packages do you use the most a...
    17                                                      
    18     Has anyone dealt with such a problem statement...
    19                                                      
    20                                                      
    21     So, I'm trying to build playlists based on val...
    22      My intents are to analyze the results with Ex...
    23                                                      
    24     Does anyone have experience in using either pl...
    25     Since I started as a data scientist, I have be...
    26                                                      
    27     Good Afternoon Everyone,&amp;#x200B;I was work...
    28     Hi all, this is a followup on [Separated from ...
    29     This is maybe not a specific DS question, but ...
                                 ...                        
    953    Specifically, as AI gets better and better, an...
    954    What is the difference between sklearn.impute....
    955                                                     
    956    ', 'author_fullname': 't2_pqifw', 'saved': Fal...
    957    I have a prospective client who’s keen to do s...
    958                                                     
    959                                                     
    960    Hello all!I have a final interview for a Sales...
    961                                                     
    962    So here’s a little about me. I’ve been a lead ...
    963    Please shoo me away to the proper sub if I'm a...
    964                                                     
    965    What's the best open source (i.e., free) appro...
    966    Bayesian Network is a probabilistic graphical ...
    967    ', 'author_fullname': 't2_r3q3m', 'saved': Fal...
    968                                                     
    969                                                     
    970    Hi, guys. I have a dataset of different addres...
    971    I have been reading a lot of quora answers and...
    972    This is my first kernel on Kaggle doing some d...
    973    Hi Guys, I need some advise or personal experi...
    974                                                     
    975    I'm finding myself in a position where I may h...
    976    I'm looking to make some data science projects...
    977    Hi, this is my first post ever, so sorry in ad...
    978    Cheers everyone! This is my first kernel on Ka...
    979    Hello /r/datascience. TLDR: given the current ...
    980    What data science course you studied from and ...
    981                                                     
    982    I'm looking for a ISO file of a distro that it...
    Name: body, Length: 983, dtype: object




```python
# Add target column for later combination
df_ds['subreddit_target'] = 1
```


```python
# Check out the nulls
df_ds.isnull().sum().sort_values()
```




    data                0
    kind                0
    body                0
    title               0
    subreddit_target    0
    dtype: int64




```python
# Same process of pulling out body/post for df_stats
try:
    df_stats['body'] = pd.Series(re.findall('(?<=selftext).{4}(.*).{4}(?=author_fullname)', df_stats['data'][0]))[0]
except:
    df_stats['body'] = ''
```


```python
# To pull out the body of the post and make it a new column
for i in range(0, df_stats.shape[0]):
    try:
        df_stats['body'][i] = re.findall('(?<=selftext).{4}(.*).{4}(?=author_fullname)', df_stats['data'][i])[0]
    except:
        df_stats['body'] = ''
```


```python
# For some reason, wrapping it in pd.Series makes this work before I loop it
try:
    df_stats['title'] = pd.Series(re.findall('(?<= .title).{4}(.*).{4}(?=link_flair_richtext)', df_stats['data'][0]))[0]
except:
    df_stats['title'] = ''
```


```python
# To pull out the title of the post and make it a new column
for i in range(0, df_stats.shape[0]):
    try:
        df_stats['title'][i] = re.findall('(?<= .title).{4}(.*).{4}(?=link_flair_richtext)', df_stats['data'][i])[0]
    except:
        df_stats['title'] = ''
```


```python
# Looks like the body got pulled in as a list
# restricting how I clean it up, turning it into a str
# Leftover
# for w in range(0,df_stats['body'].shape[0]):
#     df_stats['body'][w] = str(df_stats['body'][w])
```


```python
# Additional Clean-up - DS
df_stats['body'] = df_stats['body'].map(lambda x: x.replace('\\n',''))
df_stats['body'] = df_stats['body'].map(lambda x: x.replace('\n',''))
df_stats['body'] = df_stats['body'].map(lambda x: x.replace('\\',''))
df_stats['body'] = df_stats['body'].map(lambda x: x.replace("\\'","'"))
# df_stats['body'] = [w.replace('/n', '') for w in df_stats['body']]
```


```python
df_stats['subreddit_target'] = 0
```


```python
# Check out the nulls
df_stats.isnull().sum().sort_values()
```




    data                0
    kind                0
    body                0
    title               0
    subreddit_target    0
    dtype: int64




```python
# Renaming the columns so they're easier to discern
# Left over from previous way of solving
# df_ds.columns = ['data','kind','body_ds','title_ds']
# df_stats.columns = ['data','kind','body_stats','title_stats']
```


```python
df_ds.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>kind</th>
      <th>body</th>
      <th>title</th>
      <th>subreddit_target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>The Mod Team has decided that it would be nice...</td>
      <td>DS Book Suggestions/Recommendations Megathread</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create combined list for later usage
dflist = [df_ds, df_stats]
dfCombined = pd.concat(dflist, axis=0, sort=True)
```


```python
dfCombined.head()
# .fillna(value=" ")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>body</th>
      <th>data</th>
      <th>kind</th>
      <th>subreddit_target</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Mod Team has decided that it would be nice...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>DS Book Suggestions/Recommendations Megathread</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Welcome to this week's 'Entering &amp;amp; Transi...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Weekly 'Entering &amp;amp; Transitioning' Thread. ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Mo Data, Mo Problems. Everyone always talks ab...</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Make “Fairness by Design” Part of Machine Lear...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I'm working on making a list of Machine Learni...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Papers with Code</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check length is what I expected
dfCombined['body'].shape[0]
```




    1961




```python
dfCombined['title_body'] = dfCombined['body'] + dfCombined['title']
```


```python
dfCombined
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>body</th>
      <th>data</th>
      <th>kind</th>
      <th>subreddit_target</th>
      <th>title</th>
      <th>title_body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Mod Team has decided that it would be nice...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>DS Book Suggestions/Recommendations Megathread</td>
      <td>The Mod Team has decided that it would be nice...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Welcome to this week's 'Entering &amp;amp; Transi...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Weekly 'Entering &amp;amp; Transitioning' Thread. ...</td>
      <td>Welcome to this week's 'Entering &amp;amp; Transi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Mo Data, Mo Problems. Everyone always talks ab...</td>
      <td>Mo Data, Mo Problems. Everyone always talks ab...</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Make “Fairness by Design” Part of Machine Lear...</td>
      <td>Make “Fairness by Design” Part of Machine Lear...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I'm working on making a list of Machine Learni...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Papers with Code</td>
      <td>I'm working on making a list of Machine Learni...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>I do most of my work in Python. Building the m...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Looking for resources to learn how to launch m...</td>
      <td>I do most of my work in Python. Building the m...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[Project Link](https://github.com/HiteshGorana...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>DataScience365 ( A project started recently to...</td>
      <td>[Project Link](https://github.com/HiteshGorana...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Before I got hired, my company had a contracto...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Anyone have experience parsing hospital data f...</td>
      <td>Before I got hired, my company had a contracto...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>I'm looking for an open-source web-based tool ...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Open Source Tools for Dashboard Design</td>
      <td>I'm looking for an open-source web-based tool ...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>I've been reading around online a bit as to wh...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>MS online vs in-person</td>
      <td>I've been reading around online a bit as to wh...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>I am new to time series data, so bear with me....</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Best method for predicting the likelihood of a...</td>
      <td>I am new to time series data, so bear with me....</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Very low cost cloud GPU instances (&amp;lt;$0.15/h...</td>
      <td>Very low cost cloud GPU instances (&amp;lt;$0.15/h...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Hey all, Do people have recommendations for pi...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Pipeline Versioning (Open Source / Free) What ...</td>
      <td>Hey all, Do people have recommendations for pi...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>I am quite old (23), but would like to become ...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Data Science and being a Quant: how transferab...</td>
      <td>I am quite old (23), but would like to become ...</td>
    </tr>
    <tr>
      <th>14</th>
      <td></td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Data Democratization - Data and Analytics Take...</td>
      <td>Data Democratization - Data and Analytics Take...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>I know that python and R are the standard lang...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Mathematica is the best tool for data science ...</td>
      <td>I know that python and R are the standard lang...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Which tools and packages do you use the most a...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>What tools do you actually use at work?</td>
      <td>Which tools and packages do you use the most a...</td>
    </tr>
    <tr>
      <th>17</th>
      <td></td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Feature engineering that exploit symmetries ca...</td>
      <td>Feature engineering that exploit symmetries ca...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Has anyone dealt with such a problem statement...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>R clustering with maximum size per cluster</td>
      <td>Has anyone dealt with such a problem statement...</td>
    </tr>
    <tr>
      <th>19</th>
      <td></td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Get free GPU for training with Google Colab - ...</td>
      <td>Get free GPU for training with Google Colab - ...</td>
    </tr>
    <tr>
      <th>20</th>
      <td></td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>[Cheat Sheet] Snippets for Plotting With ggplot</td>
      <td>[Cheat Sheet] Snippets for Plotting With ggplot</td>
    </tr>
    <tr>
      <th>21</th>
      <td>So, I'm trying to build playlists based on val...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>How to use recommender Systems with Multiple "...</td>
      <td>So, I'm trying to build playlists based on val...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>My intents are to analyze the results with Ex...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Please Take This Survey if You're a College Gr...</td>
      <td>My intents are to analyze the results with Ex...</td>
    </tr>
    <tr>
      <th>23</th>
      <td></td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>How useful is a reference letter from an econ ...</td>
      <td>How useful is a reference letter from an econ ...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Does anyone have experience in using either pl...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>H2O.ai vs Datarobot? Your take</td>
      <td>Does anyone have experience in using either pl...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Since I started as a data scientist, I have be...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Are independent research papers useful for a d...</td>
      <td>Since I started as a data scientist, I have be...</td>
    </tr>
    <tr>
      <th>26</th>
      <td></td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Super helpful cheat sheets for Keras, Numpy, P...</td>
      <td>Super helpful cheat sheets for Keras, Numpy, P...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Good Afternoon Everyone,&amp;amp;#x200B;I was work...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Correlation Plot of a correlation matrix ( usi...</td>
      <td>Good Afternoon Everyone,&amp;amp;#x200B;I was work...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Hi all, this is a followup on [Separated from ...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>Step down from Data Scientist in next job- how...</td>
      <td>Hi all, this is a followup on [Separated from ...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>This is maybe not a specific DS question, but ...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'datasc...</td>
      <td>t3</td>
      <td>1</td>
      <td>How do you deal with post-job-interview though...</td>
      <td>This is maybe not a specific DS question, but ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>948</th>
      <td>Hello all. I'm a grad school student who ended...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Need to Learn How to Use SPSS Syntax ASAP</td>
      <td>Hello all. I'm a grad school student who ended...</td>
    </tr>
    <tr>
      <th>949</th>
      <td>I have been reading the Wikipedia explanations...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>ELI5: bray curtis dissimilarity matrix and UPG...</td>
      <td>I have been reading the Wikipedia explanations...</td>
    </tr>
    <tr>
      <th>950</th>
      <td>Hello all.u200bThe survey: Our survey asks peo...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Weighting an online survey with a lot of unknowns</td>
      <td>Hello all.u200bThe survey: Our survey asks peo...</td>
    </tr>
    <tr>
      <th>951</th>
      <td>Hi everyone. I'm curious whether anyone knows ...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Textbooks in statistics with great problem sets</td>
      <td>Hi everyone. I'm curious whether anyone knows ...</td>
    </tr>
    <tr>
      <th>952</th>
      <td>I am analyzing dyadic data in a multilevel mod...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Residuals plot: Is this autocorrelation?</td>
      <td>I am analyzing dyadic data in a multilevel mod...</td>
    </tr>
    <tr>
      <th>953</th>
      <td>How do you apply the Bonferroni correction if ...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Bonferroni corrections</td>
      <td>How do you apply the Bonferroni correction if ...</td>
    </tr>
    <tr>
      <th>954</th>
      <td>Hello everyone, I'm looking for books which ta...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Resources for undergrad material in Python &amp;am...</td>
      <td>Hello everyone, I'm looking for books which ta...</td>
    </tr>
    <tr>
      <th>955</th>
      <td>Hi there! I was hoping someone may be able to ...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Unsure which test to use</td>
      <td>Hi there! I was hoping someone may be able to ...</td>
    </tr>
    <tr>
      <th>956</th>
      <td>I should preface this by saying I know very li...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Help with normalization of data</td>
      <td>I should preface this by saying I know very li...</td>
    </tr>
    <tr>
      <th>957</th>
      <td>Hey r/statistics, I need some advice on how to...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Advice on an epidemiology dataset</td>
      <td>Hey r/statistics, I need some advice on how to...</td>
    </tr>
    <tr>
      <th>958</th>
      <td>I'm facing 3 problems in my current analysis (...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Groupsize differences, unequal genders and g p...</td>
      <td>I'm facing 3 problems in my current analysis (...</td>
    </tr>
    <tr>
      <th>959</th>
      <td>I am working with panel data with n=30 and t=7...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>How to interpret counterintuitive signs from m...</td>
      <td>I am working with panel data with n=30 and t=7...</td>
    </tr>
    <tr>
      <th>960</th>
      <td>I just finished gelmans Bayesian data analysis...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Where to go after Gelman's BDA3?</td>
      <td>I just finished gelmans Bayesian data analysis...</td>
    </tr>
    <tr>
      <th>961</th>
      <td>Ignore for a moment the issues with NHST.If a ...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>If you are working in the paradigm of NHST, wh...</td>
      <td>Ignore for a moment the issues with NHST.If a ...</td>
    </tr>
    <tr>
      <th>962</th>
      <td>For the “big” study this group says they hypot...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>How can I use pilot data to plan sample sizes ...</td>
      <td>For the “big” study this group says they hypot...</td>
    </tr>
    <tr>
      <th>963</th>
      <td>Hi. I need to write two predictive supply and ...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Predictive supply and demand model</td>
      <td>Hi. I need to write two predictive supply and ...</td>
    </tr>
    <tr>
      <th>964</th>
      <td>An illustration of my issue: For e.g. X is a h...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Determining which variable is more affected</td>
      <td>An illustration of my issue: For e.g. X is a h...</td>
    </tr>
    <tr>
      <th>965</th>
      <td>A group of students takes a PRE test with 50 q...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Repeat measures t-test on exam data, but pre a...</td>
      <td>A group of students takes a PRE test with 50 q...</td>
    </tr>
    <tr>
      <th>966</th>
      <td>Trying to figure out that if I have 7 variable...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Easy question from one confused boi; 7 variabl...</td>
      <td>Trying to figure out that if I have 7 variable...</td>
    </tr>
    <tr>
      <th>967</th>
      <td>Hello,I’ve been doing some analysis regardin...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>How to deal with the log of a variable where s...</td>
      <td>Hello,I’ve been doing some analysis regardin...</td>
    </tr>
    <tr>
      <th>968</th>
      <td>Hi there, I'm a bit confused about usage of F...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Questions about Firth logistic regressions</td>
      <td>Hi there, I'm a bit confused about usage of F...</td>
    </tr>
    <tr>
      <th>969</th>
      <td>I have ranked preference data for 7 items. How...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Analyzing Ranked Preference Data</td>
      <td>I have ranked preference data for 7 items. How...</td>
    </tr>
    <tr>
      <th>970</th>
      <td>I am measuring the effect of scale on the numb...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Wondering which test to conduct and how to con...</td>
      <td>I am measuring the effect of scale on the numb...</td>
    </tr>
    <tr>
      <th>971</th>
      <td>I'm looking at some instruction/examples on A/...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>[Q] non-parametric, permutations A/B testing</td>
      <td>I'm looking at some instruction/examples on A/...</td>
    </tr>
    <tr>
      <th>972</th>
      <td>&amp;amp;#x200B;</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>What is a good tutorial for learning how to ca...</td>
      <td>&amp;amp;#x200B;What is a good tutorial for learni...</td>
    </tr>
    <tr>
      <th>973</th>
      <td>&amp;amp;#x200B;</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>i'm a psych phd student who wants to befriend ...</td>
      <td>&amp;amp;#x200B;i'm a psych phd student who wants ...</td>
    </tr>
    <tr>
      <th>974</th>
      <td>Howdy, So I’m in the beginnings of a PhD in ep...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Any grad students from other fields also looki...</td>
      <td>Howdy, So I’m in the beginnings of a PhD in ep...</td>
    </tr>
    <tr>
      <th>975</th>
      <td>Correlation And Causation By Examplehttp://blo...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Correlation And Causation By Example</td>
      <td>Correlation And Causation By Examplehttp://blo...</td>
    </tr>
    <tr>
      <th>976</th>
      <td>Hi all,&amp;amp;#x200B;I'm having a bit of trouble...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>Merging item responses into a single variable ...</td>
      <td>Hi all,&amp;amp;#x200B;I'm having a bit of trouble...</td>
    </tr>
    <tr>
      <th>977</th>
      <td>Can somebody help this statistics rookie?Resea...</td>
      <td>{'approved_at_utc': None, 'subreddit': 'statis...</td>
      <td>t3</td>
      <td>0</td>
      <td>[Question] Should I use a Two-way ANOVA?</td>
      <td>Can somebody help this statistics rookie?Resea...</td>
    </tr>
  </tbody>
</table>
<p>1961 rows × 6 columns</p>
</div>




```python
# Save the cleaned-up product on the side
dfCombined.to_csv('Combined.csv', index = False)
```

## NLP

#### Use `CountVectorizer` or `TfidfVectorizer` from scikit-learn to create features from the thread titles and descriptions (NOTE: Not all threads have a description)
- Examine using count or binary features in the model
- Re-evaluate your models using these. Does this improve the model performance? 
- What text features are the most valuable? 

# N-grams = 1


```python
# Going back after the fact to add some obvious stop words
# This was form a 'normal' run of CountVectorizer, e.g. n-grams = 1
# amp seems to be some bad html code that got pulled in mistakenly
new_stop_words = {'science', 'like', 'https', 'com', 've', '10', '12', 'amp'}
stop_words = ENGLISH_STOP_WORDS.union(new_stop_words)
```


```python
# Instantiate
cvec = CountVectorizer(stop_words=stop_words) # First run through of n-grams = 1
```


```python
# Set variables and train_test_split
# Sticking with the normal 75/25 split
X = dfCombined['title_body'].values
y = dfCombined['subreddit_target']
# .map({'statistics':0, 'datascience':1})

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=42)
```


```python
# Fit and transform
cvec.fit(X_train)
X_train_transform = cvec.transform(X_train)
X_test_transform = cvec.transform(X_test)
```


```python
df_view_stats = pd.DataFrame(X_test_transform.todense(), 
                             columns=cvec.get_feature_names(),
                             index=y_test.index)
df_view_stats.head()
# .T.sort_values('statistics', ascending=False).head(10).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>00</th>
      <th>000</th>
      <th>0005</th>
      <th>0016</th>
      <th>0031</th>
      <th>004</th>
      <th>004100341sig</th>
      <th>00411621sig</th>
      <th>004p2</th>
      <th>00625</th>
      <th>...</th>
      <th>zipper</th>
      <th>zippers</th>
      <th>zjt</th>
      <th>zones</th>
      <th>zoo</th>
      <th>zuckerberg</th>
      <th>zwitch</th>
      <th>zziz</th>
      <th>µᵢ</th>
      <th>χ2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>113</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>572</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>450</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>383</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>506</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 8621 columns</p>
</div>




```python
# Most commonly used words on data science
# This was run multiple times for different sets of n-grams
word_count_test = pd.concat([df_view_stats, y_test], axis=1)
word_count_test['subreddit_target'] = word_count_test['subreddit_target'].map({0:'statistics', 1:'datascience'})
word_count_test.groupby(by='subreddit_target').sum().sort_values(by='datascience', axis=1, ascending=False).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>subreddit_target</th>
      <th>datascience</th>
      <th>statistics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>data</th>
      <td>435</td>
      <td>74</td>
    </tr>
    <tr>
      <th>learning</th>
      <td>97</td>
      <td>17</td>
    </tr>
    <tr>
      <th>work</th>
      <td>79</td>
      <td>10</td>
    </tr>
    <tr>
      <th>time</th>
      <td>74</td>
      <td>36</td>
    </tr>
    <tr>
      <th>python</th>
      <td>70</td>
      <td>5</td>
    </tr>
    <tr>
      <th>model</th>
      <td>67</td>
      <td>34</td>
    </tr>
    <tr>
      <th>know</th>
      <td>65</td>
      <td>25</td>
    </tr>
    <tr>
      <th>false</th>
      <td>64</td>
      <td>1</td>
    </tr>
    <tr>
      <th>using</th>
      <td>63</td>
      <td>16</td>
    </tr>
    <tr>
      <th>use</th>
      <td>61</td>
      <td>28</td>
    </tr>
    <tr>
      <th>looking</th>
      <td>53</td>
      <td>10</td>
    </tr>
    <tr>
      <th>new</th>
      <td>52</td>
      <td>6</td>
    </tr>
    <tr>
      <th>job</th>
      <td>51</td>
      <td>8</td>
    </tr>
    <tr>
      <th>learn</th>
      <td>51</td>
      <td>5</td>
    </tr>
    <tr>
      <th>just</th>
      <td>51</td>
      <td>11</td>
    </tr>
    <tr>
      <th>dataset</th>
      <td>44</td>
      <td>5</td>
    </tr>
    <tr>
      <th>want</th>
      <td>44</td>
      <td>21</td>
    </tr>
    <tr>
      <th>tf</th>
      <td>43</td>
      <td>0</td>
    </tr>
    <tr>
      <th>need</th>
      <td>43</td>
      <td>19</td>
    </tr>
    <tr>
      <th>project</th>
      <td>42</td>
      <td>4</td>
    </tr>
    <tr>
      <th>code</th>
      <td>41</td>
      <td>1</td>
    </tr>
    <tr>
      <th>good</th>
      <td>41</td>
      <td>12</td>
    </tr>
    <tr>
      <th>projects</th>
      <td>41</td>
      <td>7</td>
    </tr>
    <tr>
      <th>way</th>
      <td>39</td>
      <td>19</td>
    </tr>
    <tr>
      <th>set</th>
      <td>39</td>
      <td>10</td>
    </tr>
    <tr>
      <th>tensorflow</th>
      <td>39</td>
      <td>0</td>
    </tr>
    <tr>
      <th>lt</th>
      <td>38</td>
      <td>25</td>
    </tr>
    <tr>
      <th>machine</th>
      <td>38</td>
      <td>4</td>
    </tr>
    <tr>
      <th>analysis</th>
      <td>38</td>
      <td>12</td>
    </tr>
    <tr>
      <th>working</th>
      <td>37</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ljung</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>classifying</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>livestream</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>classname</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>lived</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cleanly</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>classification_report</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>classical</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>claim</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>class3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>claimed</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>claims</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>clarify</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>lol</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>logs</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>logo</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>lognormal</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>logits</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>logit</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>logistics</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>clarifying</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>logical</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>logic</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>clarityhow</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>logarithms</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>logarithmicaly</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>class1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>locked</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>class2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>χ2</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>8621 rows × 2 columns</p>
</div>




```python
# Most commonly used words on statistics
# This was run multiple times for different sets of n-grams
word_count_test = pd.concat([df_view_stats, y_test], axis=1)
word_count_test['subreddit_target'] = word_count_test['subreddit_target'].map({0:'statistics', 1:'datascience'})
word_count_test.groupby(by='subreddit_target').sum().sort_values(by='statistics', axis=1, ascending=False).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>subreddit_target</th>
      <th>datascience</th>
      <th>statistics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>data</th>
      <td>435</td>
      <td>74</td>
    </tr>
    <tr>
      <th>statistics</th>
      <td>26</td>
      <td>50</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6</td>
      <td>48</td>
    </tr>
    <tr>
      <th>variables</th>
      <td>20</td>
      <td>44</td>
    </tr>
    <tr>
      <th>variable</th>
      <td>15</td>
      <td>42</td>
    </tr>
    <tr>
      <th>test</th>
      <td>15</td>
      <td>42</td>
    </tr>
    <tr>
      <th>help</th>
      <td>36</td>
      <td>41</td>
    </tr>
    <tr>
      <th>time</th>
      <td>74</td>
      <td>36</td>
    </tr>
    <tr>
      <th>regression</th>
      <td>18</td>
      <td>36</td>
    </tr>
    <tr>
      <th>model</th>
      <td>67</td>
      <td>34</td>
    </tr>
    <tr>
      <th>use</th>
      <td>61</td>
      <td>28</td>
    </tr>
    <tr>
      <th>know</th>
      <td>65</td>
      <td>25</td>
    </tr>
    <tr>
      <th>lt</th>
      <td>38</td>
      <td>25</td>
    </tr>
    <tr>
      <th>question</th>
      <td>34</td>
      <td>25</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5</td>
      <td>23</td>
    </tr>
    <tr>
      <th>different</th>
      <td>27</td>
      <td>22</td>
    </tr>
    <tr>
      <th>distribution</th>
      <td>3</td>
      <td>21</td>
    </tr>
    <tr>
      <th>x200b</th>
      <td>25</td>
      <td>21</td>
    </tr>
    <tr>
      <th>make</th>
      <td>34</td>
      <td>21</td>
    </tr>
    <tr>
      <th>want</th>
      <td>44</td>
      <td>21</td>
    </tr>
    <tr>
      <th>way</th>
      <td>39</td>
      <td>19</td>
    </tr>
    <tr>
      <th>number</th>
      <td>15</td>
      <td>19</td>
    </tr>
    <tr>
      <th>need</th>
      <td>43</td>
      <td>19</td>
    </tr>
    <tr>
      <th>09</th>
      <td>2</td>
      <td>18</td>
    </tr>
    <tr>
      <th>statistical</th>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>sample</th>
      <td>10</td>
      <td>18</td>
    </tr>
    <tr>
      <th>linear</th>
      <td>17</td>
      <td>18</td>
    </tr>
    <tr>
      <th>day</th>
      <td>15</td>
      <td>18</td>
    </tr>
    <tr>
      <th>population</th>
      <td>0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>15</th>
      <td>7</td>
      <td>17</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>fine</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>flagship</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fishermen</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>flagged</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>flag</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fizzle</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fizzbuzz</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fixing</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fix</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fivethirtyeight</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fitted</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fitness</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fit_transform</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fit2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fishing</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fischer</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>finger</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fiscal</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>firmly</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>firm</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>firing</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>firefox</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fintech</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>finnoq</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>finnish</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>finland</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>finite</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>finishes</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>finished</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>χ2</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>8621 rows × 2 columns</p>
</div>



# N-grams = 2


```python
# This was the second run of CountVectorizer, e.g. n-grams = 2
# I removed science, because I wanted to make a differentation b/w 'science' and 'data science', and also 've', b/c it was only getting picked up b/c 'I've'
# Going to leave stop words as is for n-grams = 2, aside from html crap that got pulled in
new_stop_words = {'amp', 'x200b', 'amp x200b'}
stop_words = ENGLISH_STOP_WORDS.union(new_stop_words)
```


```python
# Instantiate
cvec2 = CountVectorizer(stop_words=stop_words, ngram_range=(2,2)) #Second run through of n-grams = 2
```


```python
# Set variables and train_test_split
# Sticking with the normal 75/25 split
X = dfCombined['title_body'].values
y = dfCombined['subreddit_target']
# .map({'statistics':0, 'datascience':1})

X_train2, X_test2, y_train2, y_test2 = train_test_split(X,
                                                    y,
                                                    random_state=42)

# Fit and transform
cvec2.fit(X_train2)
X_train_transform2 = cvec2.transform(X_train2)
X_test_transform2 = cvec2.transform(X_test2)
```


```python
df_view_stats2 = pd.DataFrame(X_test_transform2.todense(), 
                             columns=cvec2.get_feature_names(),
                             index=y_test2.index)
df_view_stats2.head()
# .T.sort_values('statistics', ascending=False).head(10).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>00 00</th>
      <th>00 29s2</th>
      <th>00 9987</th>
      <th>00 cheap</th>
      <th>00 cost</th>
      <th>00 established</th>
      <th>00 mean</th>
      <th>00 primarily</th>
      <th>00 went</th>
      <th>000 10</th>
      <th>...</th>
      <th>zippers validate</th>
      <th>zjt vector</th>
      <th>zones topping</th>
      <th>zoo ggplot2</th>
      <th>zuckerberg eric</th>
      <th>zwitch mapd</th>
      <th>zziz pwcpapers</th>
      <th>µᵢ fixed</th>
      <th>χ2 05</th>
      <th>χ2 distribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>113</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>572</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>450</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>383</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>506</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43537 columns</p>
</div>




```python
# Most commonly used words on data science
word_count_test = pd.concat([df_view_stats2, y_test2], axis=1)
word_count_test['subreddit_target'] = word_count_test['subreddit_target'].map({0:'statistics', 1:'datascience'})
word_count_test.groupby(by='subreddit_target').sum().sort_values(by='datascience', axis=1, ascending=False).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>subreddit_target</th>
      <th>datascience</th>
      <th>statistics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>data science</th>
      <td>127</td>
      <td>5</td>
    </tr>
    <tr>
      <th>machine learning</th>
      <td>38</td>
      <td>2</td>
    </tr>
    <tr>
      <th>data scientist</th>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>https www</th>
      <td>26</td>
      <td>0</td>
    </tr>
    <tr>
      <th>data scientists</th>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gt lt</th>
      <td>17</td>
      <td>0</td>
    </tr>
    <tr>
      <th>tensorflow js</th>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>https github</th>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>statistical learning</th>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>github com</th>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>data analyst</th>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>kaggle com</th>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>www kaggle</th>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>time series</th>
      <td>13</td>
      <td>3</td>
    </tr>
    <tr>
      <th>https redd</th>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>data analytics</th>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>feel like</th>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>https youtu</th>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>data set</th>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>open source</th>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>greatly appreciated</th>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>linear algebra</th>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>don know</th>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>scikit learn</th>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>data analysis</th>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>work data</th>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>lt script</th>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sql queries</th>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>little bit</th>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>new data</th>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>gallery 1hbpy1w</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gallery ehcawau</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gallery ej9di3f</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gallery html</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gallery http</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gallery o45qf8o</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gallery olzrzxz</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gallery plotly</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gallery wtdpir3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gain round</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gain opinions</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gain followers</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>future timeseries</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>future performance</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>future price</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>future research</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>future researcherdon</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>future statistical</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>future thoughts</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>future time</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>future using</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>gain academic</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>future weather</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fyi data</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fyi learning</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>g1 mn</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>g2 13</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ga 90</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ga tools</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>χ2 distribution</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>43537 rows × 2 columns</p>
</div>




```python
# Most commonly used words on statistics
word_count_test = pd.concat([df_view_stats2, y_test2], axis=1)
word_count_test['subreddit_target'] = word_count_test['subreddit_target'].map({0:'statistics', 1:'datascience'})
word_count_test.groupby(by='subreddit_target').sum().sort_values(by='statistics', axis=1, ascending=False).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>subreddit_target</th>
      <th>datascience</th>
      <th>statistics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>standard deviation</th>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>linear regression</th>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>non stationary</th>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>https imgur</th>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>regression model</th>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>independent variables</th>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>don think</th>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>imgur com</th>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>make sense</th>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>data science</th>
      <td>127</td>
      <td>5</td>
    </tr>
    <tr>
      <th>things like</th>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>normally distributed</th>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>logistic regressions</th>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>prediction model</th>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>need help</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>normal distribution</th>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>hypothesis testing</th>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>capture recapture</th>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>comp sci</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>random sample</th>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>average mean</th>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>post test</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>real time</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>pre post</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>make statistical</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>data excel</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>hotspot mapping</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>independent variable</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>index variables</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>statistical curve</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>frames day</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>framework aware</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>framework building</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>framework cheersbest</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>framework consistent</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>framework guidance</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>framework implemented</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>framework interactive</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fragments feeding</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fraction discard</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>forward similar</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fpsyg 2018</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>forward want</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>forxa03xa0months july</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>foundation hiring</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>foundation mathematics</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>foundation prior</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>foundations predictive</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>foundations python</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>founder kdnuggets</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fourmilab ch</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fourth generate</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>foxes hounds</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>foxes immediately</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>foxes seven</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>foxhole inside</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fp growth</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fp persons</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>fpsyg 09</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>χ2 distribution</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>43537 rows × 2 columns</p>
</div>




```python
# Instantiate and fit
lr2 = LogisticRegression()
lr2.fit(X_train_transform2, y_train2)
lr2.score(X_train_transform2, y_train2)
```




    0.9863945578231292




```python
lr2.score(X_test_transform2, y_test2)
# Looks like a pretty decent overfit
```




    0.7637474541751528



## Predicting subreddit using Random Forests + Another Classifier


```python
# Instantiate and fit
# From here on out, it's n-grams = 2
lr = LogisticRegression()
lr.fit(X_train_transform, y_train)
lr.score(X_train_transform, y_train)
```




    0.9897959183673469




```python
lr.score(X_test_transform, y_test)
# Looks like a pretty decent overfit
```




    0.8757637474541752



#### We want to predict a binary variable - class `0` for one of your subreddits and `1` for the other.


```python
preds = lr.predict(X_test_transform)
pred_proba = lr.predict_proba(X_test_transform)[:,1]
```


```python
roc_auc = roc_auc_score(y_test, preds)
roc_auc
```




    0.8772046367954297




```python
roc_auc = roc_auc_score(y_test, preds)
FPR, TPR, thresholds = roc_curve(y_test, pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(FPR, TPR, label='Logistic Regression (area = %0.2f)' % roc_auc)
plt.title('ROC-AUC (n-grams=1)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc="lower right")
plt.show()
```


![png](/images/project_3_code_files/project_3_code_87_0.png)


#### Thought experiment: What is the baseline accuracy for this model?


```python
## I'm going to take an educated guess that the baseline accuracy is 50%, as in, randomly guessing
```

#### Create a `RandomForestClassifier` model to predict which subreddit a given post belongs to.


```python
# Instantiate
rf = RandomForestClassifier()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

#### Use cross-validation in scikit-learn to evaluate the model above. 
- Evaluate the accuracy of the model, as well as any other metrics you feel are appropriate. 
- **Bonus**: Use `GridSearchCV` with `Pipeline` to optimize your `CountVectorizer`/`TfidfVectorizer` and classification model.


```python
cvs_train = cross_val_score(rf, X_train_transform, y_train, cv=cv, n_jobs=-1)

print(cvs_train)
print(cvs_train.mean())
```

    [0.81292517 0.86734694 0.79591837 0.78231293 0.79931973]
    0.8115646258503402



```python
cvs_test = cross_val_score(rf, X_test_transform, y_test, cv=cv, n_jobs=-1)

print(cvs_test)
print(cvs_test.mean())
# Still slight overfit
```

    [0.70707071 0.71717172 0.76767677 0.82474227 0.74226804]
    0.7517859002395084


#### Repeat the model-building process using a different classifier (e.g. `MultinomialNB`, `LogisticRegression`, etc)

# MultinomialNB


```python
mnb = MultinomialNB()
```


```python
cvs_train = cross_val_score(mnb, X_train_transform, y_train, cv=cv, n_jobs=-1)

print(cvs_train)
print(cvs_train.mean())
```

    [0.82312925 0.8537415  0.81632653 0.80272109 0.81972789]
    0.8231292517006802



```python
cvs_test = cross_val_score(mnb, X_test_transform, y_test, cv=cv, n_jobs=-1)

print(cvs_test)
print(cvs_test.mean())
# Not as bad of an overfit
```

    [0.7979798  0.75757576 0.7979798  0.81443299 0.77319588]
    0.788232843902947


# GaussianNB


```python
gnb = GaussianNB()
```


```python
cvs_train = cross_val_score(gnb, X_train_transform.toarray(), y_train, cv=cv, n_jobs=-1)

print(cvs_train)
print(cvs_train.mean())
```

    [0.77891156 0.80612245 0.78231293 0.81292517 0.80952381]
    0.7979591836734693



```python
cvs_test = cross_val_score(gnb, X_test_transform.toarray(), y_test, cv=cv, n_jobs=-1)

print(cvs_test)
print(cvs_test.mean())
# Overfit isn't as much of a problem on this model
# However, the overall score isn't as strong as the other models
```

    [0.71717172 0.76767677 0.80808081 0.78350515 0.70103093]
    0.7554930750807038


# Executive Summary
---
Put your executive summary in a Markdown cell below.

    Reclassifying all of Reddit is an incredible daunting task. However, the machine learning and natural language processing abilities of Python can turn this into a manageable task. Reddit calls itself the 'frontpage of the internet,' and indicative of the innovation that drove the creation of the internet, Reddit can innovate to overcome this challenge as it has countless obstacles before this.

    Specifically, the distinction between r/DataScience and r/Statistics is relatively small as these subreddits generally discuss similar ideas and concepts. Despite these similarities, I believe my models performed quite well (especially my first run of Logistic Regression using n-grams = 1). Additionally, I chose to remove specific stop words ('science', 'https', 'com') that would more easily identify r/DataScience as the correct subreddit, in order to 'challenge' my modeling and evaluation skills, as well as allow this process to more generally be applied to all of Reddit's various subreddits. Removing these stop word would have increased my models' classifying ability even further. 
    
    Finally, I believe that this machine learning/NLP process can be applied to Reddit as a whole to help reclassify and realign it's subreddits with a high degree of success. Coupled with Reddit's strong community, including its committed mods, this is challenge that Reddit can overcome, and potentially be stronger off because of it.
