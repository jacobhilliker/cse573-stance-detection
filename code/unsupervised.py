# !pip install transformers
import string
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import BertTokenizer
import re
import pandas as pd
import numpy as np
import nltk 
from nltk.cluster import KMeansClusterer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')


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

    count = Counter(emotions)

def sentiment_analyze(cleaned_text1):
    score = sia().polarity_scores(cleaned_text1)
    if score['neg'] > score['pos']:
        # print("Negative Sentiment")
        return "neg"
    if score['neg'] < score['pos']:
        # print("Positive Sentiment")
        return "pos"
    else:
        # print("Neutral Sentiment")
        return "other"

    return sentiment_analyze(cleaned_text)

def unsupervised_cluster():

    all_files = [Dataset.TRAIN, Dataset.TEST]
    li = []
    
    for filename in all_files:
        sub_df = pd.read_csv(filename, index_col=None, header=0)
        li.append(sub_df)

    df = pd.concat(li, axis=0, ignore_index=True)

    # Create BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    max_input_length=64

    df["input_ids"] = [tokenizer.convert_tokens_to_ids(tokenize_truncate(sent, tokenizer, max_input_length)) for sent in df.Tweet.values]
    df_topics = df.Target.unique()
    topic_ids = [tokenizer.convert_tokens_to_ids(tokenize_truncate(sent, tokenizer, max_input_length)) for sent in df_topics]
    NUM_CLUSTERS = len(topic_ids)
    sentences = df.Tweet
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

    text_tweets = df.Tweet.to_list()
    length = len(text_tweets)

    df.sentiment = df.Tweet.apply(sentiment_analysis_sent)
    df = df.replace({'Sentiment': 'neg'},'AGAINST')
    df = df.replace({'Sentiment': 'pos'},'FAVOR')
    df = df.replace({'Sentiment': 'other'},'NONE')
    # return np.sum(df.Stance == df.Sentiment)/len(df), np.sum(df.Target == df.cluster_name)/len(df), np.sum(np.logical_and(df.Target == df.cluster_name, df.Stance == df.Sentiment))/len(df)

    sentiment_f1 = f1_score(df.Stance, df.Sentiment, average="weighted")
    sentiment_accuracy = accuracy_score(df.Stance, df.Sentiment)
    sentiment_precision = precision_score(df.Stance, df.Sentiment, average="weighted")
    sentiment_recall = recall_score(df.Stance, df.Sentiment, average="weighted")
    
    topic_f1 = f1_score(df.Target, df.cluster_name, average="weighted")
    topic_accuracy = accuracy_score(df.Target, df.cluster_name)
    topic_precision = precision_score(df.Target, df.cluster_name, average="weighted")
    topic_recall = recall_score(df.Target, df.cluster_name, average="weighted")

    both_f1 = f1_score(df.Stance + df.Target, df.Sentiment + df.cluster_name, average="weighted")
    both_accuracy = accuracy_score(df.Stance + df.Target, df.Sentiment + df.cluster_name, average="weighted")
    both_precision = precision_score(df.Stance + df.Target, df.Sentiment + df.cluster_name, average="weighted")
    both_recall = recall_score(df.Stance + df.Target, df.Sentiment + df.cluster_name, average="weighted")

    return json.dumps(
        [
            {
                "model": "sentiment",
                "accuracy": sentiment_accuracy,
                "precision": sentiment_precision,
                "recall": sentiment_recall,
                "f1": sentiment_f1,
            },
            {
                "model": "topic modelling",
                "accuracy": topic_accuracy,
                "precision": topic_precision,
                "recall": topic_recall,
                "f1": topic_f1,
            },
            {
                "model": "sentiment+topic modelling",
                "accuracy": both_accuracy,
                "precision": both_precision,
                "recall": both_recall,
                "f1": both_f1,
            }
        ]
        
    )