# !pip install transformers
import string
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import BertTokenizer
import re
import json
import pandas as pd
import numpy as np
import nltk 
from nltk.cluster import KMeansClusterer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

def tokenize_truncate(sentence, tokenizer, max_input_length):  
    # sentence= re.sub(r"(@[A-Za-z0–9_]+)|[^\w\s]|#|http\S+", "", sentence)
    sentence= re.sub(r"[^\w\s]|#|http\S+", "", sentence)
    tokens = tokenizer.tokenize(sentence) 
    # tokens = [tokenizer.cls_token] + tokens[:max_input_length-2] + [tokenizer.sep_token]
    tokens = tokens[:max_input_length-2]
    if len(tokens) < max_input_length-2:
      tokens+=[tokenizer.pad_token]*(max_input_length-len(tokens)-2)
    return tokens


def sentiment_analysis_sent(text):
  # # converting to lowercase
  lower_case = text.lower()

  # Removing punctuations
  cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

  # splitting text into words
  tokenized_words = word_tokenize(cleaned_text, "english")

  # Removing stop words from the tokenized words list
  final_words = []
  for word in tokenized_words:
      if word not in stopwords.words('english'):
          final_words.append(word)

  # Get emotions text
  emotions = []
  with open(e_name, 'r') as file:
      for line in file:
          clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
          word, emotion = clear_line.split(':')
          if word in final_words:
              emotions.append(emotion)


def sentiment_analyze(cleaned_text1):
    score = sia().polarity_scores(cleaned_text1)
    if score['neg'] > score['pos']:
        # print("Negative sentiment")
        return "neg"
    if score['neg'] < score['pos']:
        # print("Positive sentiment")
        return "pos"
    else:
        # print("Neutral sentiment")
        return "other"

    return sentiment_analyze(cleaned_text)

def unsupervised_cluster():

    # , 'data/semeval2016_corrected_test.csv']
    # li = []
    
    # for filename in all_files:
    #     sub_df = pd.read_csv(filename, index_col=None, header=0)
    #     li.append(sub_df)

    # df = pd.concat(li, axis=0, ignore_index=True)

    df = pd.read_csv('data/semeval2016_corrected.csv')
    # Create BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    max_input_length=64

    df["input_ids"] = [tokenizer.convert_tokens_to_ids(tokenize_truncate(sent, tokenizer, max_input_length)) for sent in df.tweet.values]
    df_topics = df.topic.unique()
    topic_ids = [tokenizer.convert_tokens_to_ids(tokenize_truncate(sent, tokenizer, max_input_length)) for sent in df_topics]
    NUM_CLUSTERS = len(topic_ids)
    X = np.array(df['input_ids'].tolist())

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=NUM_CLUSTERS,init='random')
    kmeans.fit(topic_ids)
    Z = kmeans.predict(X)

    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=1,avoid_empty_clusters=True)

    topic_assigned_clusters = kclusterer.cluster(topic_ids, assign_clusters=True)

    # topic_df.columns = ["cluster"]

    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    df['cluster'] = pd.Series(assigned_clusters, index=df.index)
    # df['centroid'] = df['cluster'].apply(lambda x: kclusterer.means()[x]) 

    topic_dict = {}
    for i in range(NUM_CLUSTERS):
        topic_dict.update({topic_assigned_clusters[i]: df_topics[i]})
    topic_df = pd.DataFrame.from_dict(topic_dict, orient='index')

    topic_df.columns = ["cluster_name"]
    # topic_df.reset_index(inplace=True)
    topic_df = topic_df.rename(columns = {'index':'cluster'})

    df = df.join(topic_df, on="cluster", lsuffix="_l")
    #### TOPIC MODELLING COMPLETE

    e_name = 'emotion.txt'

    text_tweets = df.tweet.to_list()
    length = len(text_tweets)

    df.sentiment = df.tweet.apply(sentiment_analysis_sent)
    df = df.replace({'sentiment': 'neg'},'AGAINST')
    df = df.replace({'sentiment': 'pos'},'FAVOR')
    df = df.replace({'sentiment': 'other'},'NONE')
    # return np.sum(df.stance == df.sentiment)/len(df), np.sum(df.topic == df.cluster_name)/len(df), np.sum(np.logical_and(df.topic == df.cluster_name, df.stance == df.sentiment))/len(df)

    sentiment_f1 = f1_score(df.stance, df.sentiment, average="weighted")
    sentiment_accuracy = accuracy_score(df.stance, df.sentiment)
    sentiment_precision = precision_score(df.stance, df.sentiment, average="weighted")
    sentiment_recall = recall_score(df.stance, df.sentiment, average="weighted")
    
    topic_f1 = f1_score(df.topic, df.cluster_name, average="weighted")
    topic_accuracy = accuracy_score(df.topic, df.cluster_name)
    topic_precision = precision_score(df.topic, df.cluster_name, average="weighted")
    topic_recall = recall_score(df.topic, df.cluster_name, average="weighted")

    both_f1 = f1_score(df.stance + df.topic, df.sentiment + df.cluster_name, average="weighted")
    both_accuracy = accuracy_score(df.stance + df.topic, df.sentiment + df.cluster_name)
    both_precision = precision_score(df.stance + df.topic, df.sentiment + df.cluster_name, average="weighted")
    both_recall = recall_score(df.stance + df.topic, df.sentiment + df.cluster_name, average="weighted")

    return json.dumps(
        [
            json.dumps({
                "model": "sentiment",
                "accuracy": sentiment_accuracy,
                "precision": sentiment_precision,
                "recall": sentiment_recall,
                "f1": sentiment_f1,
            }),
            json.dumps({
                "model": "topic_modeling",
                "accuracy": topic_accuracy,
                "precision": topic_precision,
                "recall": topic_recall,
                "f1": topic_f1,
            }),
            json.dumps({
                "model": "sentiment_and_topic_modeling",
                "accuracy": both_accuracy,
                "precision": both_precision,
                "recall": both_recall,
                "f1": both_f1,
            })
        ]
        
    )
def report_unsupervised_results():
    return json.dumps([
            json.dumps({
                "model": "sentiment", 
                "accuracy": 0.44371997254632806, 
                "precision": 0.45223196282651573, 
                "recall": 0.44371997254632806,
                 "f1": 0.41406603802107156}),
            json.dumps({
                "model": "topic_modeling", 
                "accuracy": 0.20212765957446807, 
                "precision": 0.22022955216809206, 
                "recall": 0.20212765957446807, 
                "f1": 0.20466423147977636}), 
            json.dumps({
                "model": "sentiment_and_topic_modeling", 
                "accuracy": 0.10638297872340426,
                "precision": 0.13865858135268772, 
                "recall": 0.10638297872340426, 
                "f1": 0.11095636130466122})
            ])

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    unsupervised_cluster()