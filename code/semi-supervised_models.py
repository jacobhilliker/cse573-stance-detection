from supervised_models import Learning
from process_topic_data import Dataset
from supervised_models import train_models
from process_data import get_stances, get_tweets, get_vectors, vectorize
from supervised_models import train_fastText, train_knn, train_naive_bayes, train_svm
from process_topic_data import Stance
from process_topic_data import Topic
from process_topic_data import (
    get_abortion_indices,
    get_atheism_indices,
    get_climate_indices,
    get_feminism_indices,
    get_hrc_indices,
)
from process_topic_data import load_data
from stance_topic_data import get_stance

def get_regex_filter(cur_topic):
    if cur_topic==Topic.ATHEISM:
        return "Atheism"
    elif cur_topic==Topic.HRC:
        return "Hrc"
    elif cur_topic==Topic.CLIMATE:
        return "Climate"
    elif cur_topic==Topic.ABORTION:
        return "Abortion"
    else:
        return "Feminism"


def semisup():
    train_data = load_data(Dataset.TRAIN)

    abortion_indices = get_abortion_indices(train_data)
    atheism_indices = get_atheism_indices(train_data)
    climate_indices = get_climate_indices(train_data)
    feminism_indices = get_feminism_indices(train_data)
    hrc_indices = get_hrc_indices(train_data)

    for topic_indices in [
        abortion_indices,
        atheism_indices,
        climate_indices,
        feminism_indices,
        hrc_indices,
    ]:

        for topic_index in topic_indices:
            datum = train_data[topic_index]
            tweet = datum["tweet"]
            topic = datum["topic"]
            #Breakdown:
                #1) Use the topic index to fetch which regex patterns to use
                #2) Implement the regex matching logic e.g. for bible verses 23:11 etc
                #3) What if some tweets contain both the FOR and AGAINST patterns? Just mark it neutral if equal no patterns match else pick the max one
                #4) Now we use the labels to mark the train data and run the training model
                #5) Repeat above steps for test data and calculate the accuracy.

            stance_label = get_stance(tweet, topic)

            datum["stance"] = stance_label

    print("SemiSupervised: ")
    train_models(train_data + load_data(Dataset.TEST), Learning.SEMISUPERVISED)

if __name__ == "__main__":
    semisup()