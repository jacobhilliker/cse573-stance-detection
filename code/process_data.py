"""
Author: Jacob Hilliker and Jack Myers
Purpose: Builds data structure out of SemEval-2016 text file
"""

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd


def get_attribute(data, attribute):
    """
    Parameters:
      data - list of dictionaries
      attribute - one of 'topic', 'tweet', 'stance', 'vector'
    Returns a list of the desired attribute"""

    return [datum[attribute] for datum in data]


def get_topics(data):
    """
    Returns list of topics"""
    return get_attribute(data, "topic")


def get_tweets(data):
    """
    Returns list of Tweets"""
    return get_attribute(data, "tweet")


def get_stances(data):
    """
    Returns list of stance labels"""
    return get_attribute(data, "stance")


def get_vectors(data):
    """
    Returns list of vectorized Tweets
    """
    return get_attribute(data, "vector")


def split_data():
    """
    Split dataset into persisting training and testing datasets
    """

    data = pd.read_csv("data/semeval2016_corrected.txt")

    data = data.sample(frac=1).reset_index(drop=True)  # randomize data

    # use first 0.75 for training data, and last 0.25 for test
    split_point = int(data.shape[0] * 3 / 4)

    data[:split_point].to_csv("data/semeval2016_corrected_train.csv", index=False)
    data[split_point:].to_csv("data/semeval2016_corrected_test.csv", index=False)

    return


def load_corrected_data(filePath):
    """
    Returns a list of dictionaries, where each entry is of the form ['topic', 'tweet', 'stance', 'vector'].
    """
    data = pd.read_csv(filePath)

    tweets = [
        {
            "topic": datum["topic"],
            "tweet": datum["tweet"],
            "stance": datum["stance"],
            "vector": None,
        }
        for datum in data
    ]

    return tweets


def load_raw_data():
    """
    Returns a list of dictionaries containing Tweet information following the format above.
    """

    tweets = []
    data = open("data/semeval2016_full.txt")

    # Strip column labels
    _ = data.readline()

    # Break text into topic, Tweet, and stance
    for line in data.readlines():

        current_topic = ""
        current_tweet = ""
        current_stance = ""

        # Splits each row into [ID, Topic, Tweet, Stance]
        split_line = line.split("\t")

        # Parse topic and tweet
        current_topic = split_line[1]
        current_tweet = split_line[2]

        # Trim SemST hashtag
        current_tweet = current_tweet[: len(current_tweet) - 7]

        # Parse stance and remove newline
        current_stance = split_line[3].strip()

        # Build dictionary to add to list
        current_dict = {
            "topic": current_topic,
            "tweet": current_tweet,
            "stance": current_stance,
            "vector": None,
        }

        tweets.append(current_dict)

    return tweets


def vectorize(data):
    """
    Adds the TFIDF vector representation for each Tweet
    """

    # Build list of just the tweets
    tweets = [datum["tweet"] for datum in data]

    vectorizer = TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
    tweet_vectors = vectorizer.fit_transform(tweets).toarray()

    for i, datum in enumerate(data):
        datum["vector"] = tweet_vectors[i]

    return data


if __name__ == "__main__":
    tweets = vectorize(load_corrected_data("data/semeval2016_corrected.txt"))
    print(tweets[0]["tweet"])
    print(tweets[0]["vector"])
