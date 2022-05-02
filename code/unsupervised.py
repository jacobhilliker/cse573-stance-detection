# packages to store and manipulate data
import json
import pandas as pd
import numpy as np

# plotting packages
import matplotlib.pyplot as plt
from sklearn.metrics import *
# import seaborn as sns

# model building package
import sklearn

# package to clean text
import re

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import string
from collections import Counter
# import GetOldTweets3 as got
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from transformers import BertTokenizer


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'



def find_mentioned(tweet):
    '''This function will extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)  

def find_hashtags(tweet):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)  

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
  with open('code/emotion.txt', 'r') as file:
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

def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet

def remove_users(tweet):
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at
    return tweet

def clean_tweet(tweet, bigrams=False):
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
    tweet = re.sub('\s+', ' ', tweet) #remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
    tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords] # remove stopwords

    tweet_token_list = [word_rooter(word) if '#' not in word else word
                        for word in tweet_token_list] # apply word rooter
    # if bigrams:
    #     tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
    #                                         for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet

def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %s words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        # topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
        #                 for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

def unsupervised_cluster():

    # , 'data/semeval2016_corrected_test.csv']
    # li = []
    
    # for filename in all_files:
    #     sub_df = pd.read_csv(filename, index_col=None, header=0)
    #     li.append(sub_df)

    # df = pd.concat(li, axis=0, ignore_index=True)

    df = pd.read_csv('data/semeval_train.csv')
    # Create BERT tokenizer
    # create dataframe where each use of hashtag gets its own row
     
    df['mentioned'] = df.Tweet.apply(find_mentioned)
    df['hashtags'] = df.Tweet.apply(find_hashtags)
    # take the rows from the hashtag columns where there are actually hashtags
    hashtags_list_df = df.loc[
                        df.hashtags.apply(
                            lambda hashtags_list: hashtags_list !=[]
                        ),['hashtags']]
                        
    flattened_hashtags_df = pd.DataFrame(
        [hashtag for hashtags_list in hashtags_list_df.hashtags
        for hashtag in hashtags_list],
        columns=['hashtag'])
    # number of unique hashtags
    flattened_hashtags_df['hashtag'].unique().size
    # count of appearances of each hashtag
    popular_hashtags = flattened_hashtags_df.groupby('hashtag').size()\
                                            .reset_index(name='counts')\
                                            .sort_values('counts', ascending=False)\
                                            .reset_index(drop=True)
                                            # number of times each hashtag appears
    counts = flattened_hashtags_df.groupby(['hashtag']).size()\
                                .reset_index(name='counts')\
                                .counts

    # UNCOMMENT TO SEE distribution of hashtags
    # # define bins for histogram                              
    # my_bins = np.arange(0,counts.max()+2, 5)-0.5

    # # plot histogram of tweet counts
    # plt.figure()
    # plt.hist(counts, bins = my_bins)
    # plt.xlabels = np.arange(1,counts.max()+1, 1)
    # plt.xlabel('hashtag number of appearances')
    # plt.ylabel('frequency')
    # plt.yscale('log', nonposy='clip')
    # plt.show()

    # take hashtags which appear at least this amount of times
    min_appearance = 10
    # find popular hashtags - make into python set for efficiency
    popular_hashtags_set = set(popular_hashtags[
                            popular_hashtags.counts>=min_appearance
                            ]['hashtag'])
    # make a new column with only the popular hashtags
    hashtags_list_df['popular_hashtags'] = hashtags_list_df.hashtags.apply(
                lambda hashtag_list: [hashtag for hashtag in hashtag_list
                                    if hashtag in popular_hashtags_set])
    # drop rows without popular hashtag
    popular_hashtags_list_df = hashtags_list_df.loc[
                hashtags_list_df.popular_hashtags.apply(lambda hashtag_list: hashtag_list !=[])]

    # make new dataframe
    hashtag_vector_df = popular_hashtags_list_df.loc[:, ['popular_hashtags']]

    for hashtag in popular_hashtags_set:
        # make columns to encode presence of hashtags
        hashtag_vector_df['{}'.format(hashtag)] = hashtag_vector_df.popular_hashtags.apply(
            lambda hashtag_list: int(hashtag in hashtag_list))
        
    hashtag_matrix = hashtag_vector_df.drop('popular_hashtags', axis=1)

    # calculate the correlation matrix
    correlations = hashtag_matrix.corr()

    # # plot the correlation matrix
    # plt.figure(figsize=(10,10))
    # sns.heatmap(correlations,
    #     cmap='RdBu',
    #     vmin=-1,
    #     vmax=1,
    #     square = True,
    #     cbar_kws={'label':'correlation'})
    # plt.show()

    my_stopwords = nltk.corpus.stopwords.words('english')
    word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
    
    df['clean_tweet'] = df.Tweet.apply(clean_tweet)


    # the vectorizer object will be used to transform text to vector form
    vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')

    # apply transformation
    tf = vectorizer.fit_transform(df['clean_tweet']).toarray()

    # tf_feature_names tells us what word each column in the matric represents
    tf_feature_names = vectorizer.get_feature_names()
    number_of_topics = 10

    model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)

    model.fit(tf)

    no_top_words = 10

    top_words_df = display_topics(model, tf_feature_names, no_top_words).transpose()
    vocabulary = top_words_df.to_numpy().flatten()

    
    # Create BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    max_input_length=10

    df["input_ids"] = [tokenizer.convert_tokens_to_ids(tokenize_truncate(sent, tokenizer, max_input_length)) for sent in df.Tweet.values]

    df_topics = df.Target.unique()
    topic_ids = [tokenizer.convert_tokens_to_ids(tokenize_truncate(sent, tokenizer, max_input_length)) for sent in df_topics]
    NUM_CLUSTERS = len(topic_ids)
    sentences = df.Tweet
    X = np.array(df['input_ids'].tolist())#.sample(n=10, random_state=1).tolist())
    # X = df['input_ids'].to_numpy()

    topic_dict = {}
    for i in range(NUM_CLUSTERS):
        topic_dict.update({i: df_topics[i]})
        
    topic_df = pd.DataFrame.from_dict(topic_dict, orient='index')

    topic_df.columns = ["cluster_name"]
    # topic_df.reset_index(inplace=True)
    topic_df = topic_df.rename(columns = {'index':'cluster'})

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=NUM_CLUSTERS,init='random')
    kmeans.fit(topic_ids)
    df["cluster"] = kmeans.predict(X)
    topic_df.columns = ["Target"]

    df = df.set_index('cluster').join(topic_df, lsuffix="_l")
    # result_df = df.where(df.Target==df.Target_l).dropna()
    
    ######################## SENTI
    text_tweets = df.Tweet.to_list()
    length = len(text_tweets)

    df.sentiment = df.Tweet.apply(sentiment_analysis_sent)
    df = df.replace({'Stance': 'AGAINST'}, 'neg')
    df = df.replace({'Stance': 'FAVOR'}, 'pos')
    df = df.replace({'Stance': 'NONE'}, 'other')


    sentiment_f1 = f1_score(df.Stance, df.Sentiment, average="weighted")
    sentiment_accuracy = accuracy_score(df.Stance, df.Sentiment)
    sentiment_precision = precision_score(df.Stance, df.Sentiment, average="weighted")
    sentiment_recall = recall_score(df.Stance, df.Sentiment, average="weighted")
    
    topic_f1 = f1_score(df.Target, df.Target_l, average="weighted")
    topic_accuracy = accuracy_score(df.Target, df.Target_l)
    topic_precision = precision_score(df.Target, df.Target_l, average="weighted")
    topic_recall = recall_score(df.Target, df.Target_l, average="weighted")

    both_f1 = f1_score(df.Stance + df.Target, df.Sentiment + df.Target_l, average="weighted")
    both_accuracy = accuracy_score(df.Stance + df.Target, df.Sentiment + df.Target_l)
    both_precision = precision_score(df.Stance + df.Target, df.Sentiment + df.Target_l, average="weighted")
    both_recall = recall_score(df.Stance + df.Target, df.Sentiment + df.Target_l, average="weighted")

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
# def report_unsupervised_results():
#     return json.dumps([
#             json.dumps({
#                 "model": "sentiment", 
#                 "accuracy": 0.44371997254632806, 
#                 "precision": 0.45223196282651573, 
#                 "recall": 0.44371997254632806,
#                  "f1": 0.41406603802107156}),
#             json.dumps({
#                 "model": "topic_modeling", 
#                 "accuracy": 0.20212765957446807, 
#                 "precision": 0.22022955216809206, 
#                 "recall": 0.20212765957446807, 
#                 "f1": 0.20466423147977636}), 
#             json.dumps({
#                 "model": "sentiment_and_topic_modeling", 
#                 "accuracy": 0.10638297872340426,
#                 "precision": 0.13865858135268772, 
#                 "recall": 0.10638297872340426, 
#                 "f1": 0.11095636130466122})
#             ])

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    a = unsupervised_cluster()
    print(a)