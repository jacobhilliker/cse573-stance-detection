'''
Author: Jacob Hilliker
Purpose: Builds data structure out of SemEval-2016 text file
'''

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

NUM_TWEETS = 2914

'''
Returns a list of dictionaries, where each entry is of the form ['topic', 'tweet', 'stance', 'vector'].
'''
def load_corrected_data():

    tweets = []
    data = pd.read_csv('data/semeval2016_corrected.txt')

    for i in range(NUM_TWEETS):

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

    vectorizer = TfidfVectorizer()
    tweet_vectors = vectorizer.fit_transform(tweets)

    for i, datum in enumerate(data):
        datum['vector'] = tweet_vectors[i]

    return data

if __name__ == '__main__':
    
    tweets = vectorize(load_corrected_data())
    print(tweets[0]['tweet'])
    print(tweets[0]['vector'])