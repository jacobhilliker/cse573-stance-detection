'''
Author: Jacob Hilliker
Purpose: Builds data structure out of SemEval-2016 text file
'''

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import pandas as pd

'''
Parameters:
  data - list of dictionaries
  attribute - one of 'topic', 'tweet', 'stance', 'vector'
Returns a list of the desired attribute
'''
def get_attribute(data, attribute):
    result = []
    for i in range(len(data)):
        result.append(data[i][attribute])

    return result

'''
Returns list of topics
'''
def get_topics(data):
    return get_attribute(data, 'topic')

'''
Returns list of Tweets
'''
def get_tweets(data):
    return get_attribute(data, 'tweet')

'''
Returns list of stance labels
'''
def get_stances(data):
    return get_attribute(data, 'stance')

'''
Returns list of vectorized Tweets
'''
def get_vectors(data):
    return get_attribute(data, 'vector')


'''
Split dataset into persisting training and testing datasets
'''
def split_data():

    data = pd.read_csv('data/semeval2016_corrected.txt')

    data = data.sample(frac = 1).reset_index(drop=True) # randomize data

    # use first 0.75 for training data, and last 0.25 for test
    split_point = int(data.shape[0] * 3/4)   

    data[:split_point].to_csv('data/semeval2016_corrected_train.csv', index=False)
    data[split_point:].to_csv('data/semeval2016_corrected_test.csv', index=False)

    return

'''
Returns a list of dictionaries, where each entry is of the form ['topic', 'tweet', 'stance', 'vector'].
'''
def load_corrected_data(filePath):

    tweets = []
    data = pd.read_csv(filePath)

    for i in range(len(data)):

        current_dict = {
            'topic': data['topic'][i],
            'tweet': data['tweet'][i],
            'stance': data['stance'][i],
            'vector': None
        }

        tweets.append(current_dict)
    
    return tweets

'''
Returns a list of dictionaries containing Tweet information following the format above.
'''
def load_raw_data():

    tweets = []
    data = open('data/semeval2016_full.txt')

    # Strip column labels
    labels_unused = data.readline()

    # Break text into topic, Tweet, and stance
    for line in data.readlines():

        current_topic = ''
        current_tweet = ''
        current_stance = ''

        # Splits each row into [ID, Topic, Tweet, Stance]
        split_line = line.split('\t')

        # Parse topic and tweet
        current_topic = split_line[1]
        current_tweet = split_line[2]
        
        # Trim SemST hashtag
        current_tweet = current_tweet[:len(current_tweet) - 7]

        # Parse stance and remove newline
        current_stance = split_line[3].strip()

        # Build dictionary to add to list
        current_dict = {
            'topic': current_topic,
            'tweet': current_tweet,
            'stance': current_stance,
            'vector': None
        }

        tweets.append(current_dict)

    return tweets

'''
Adds the TFIDF vector representation for each Tweet
'''
def vectorize(data):

    # Build list of just the tweets
    tweets = []
    for datum in data:
        tweets.append(datum['tweet'])

    vectorizer = TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
    tweet_vectors = vectorizer.fit_transform(tweets).toarray()

    for i, datum in enumerate(data):
        datum['vector'] = tweet_vectors[i]

    return data

if __name__ == '__main__':
    
    tweets = vectorize(load_corrected_data('data/semeval2016_corrected.txt'))
    print(tweets[0]['tweet'])
    print(tweets[0]['vector'])