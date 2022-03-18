'''
Author: Jacob Hilliker
Purpose: Builds data structure out of SemEval-2016 text file
'''

'''
Returns a list of dictionaries containing Tweet information.
The keys of the dictionary are 'topic', 'tweet', and 'stance'.
'''
def load_data():

    tweets = []
    data = open('data/semeval2016_full.txt')

    # Strip column labels
    labels = data.readline()

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
            'stance': current_stance
        }

        tweets.append(current_dict)

    return tweets

if __name__ == '__main__':
    data = load_data()

    print(data[0]['topic'])
    print(data[0]['tweet'])
    print(data[0]['stance'])