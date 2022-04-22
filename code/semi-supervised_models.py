from supervised_models import Learning
from process_topic_data import Dataset
from supervised_models import train_models
from process_data import get_stances, get_tweets, get_vectors, vectorize
from supervised_models import train_fastText, train_knn, train_naive_bayes, train_svm
from process_topic_data import Stance
from process_topic_data import (
    get_abortion_indices,
    get_atheism_indices,
    get_climate_indices,
    get_feminism_indices,
    get_hrc_indices,
)
from process_topic_data import load_data


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
            stance_label = Stance.NONE  # TODO: use regex to label stance
            datum["stance"] = stance_label

    print("SemiSupervised: ")
    train_models(train_data + load_data(Dataset.TEST), Learning.SEMISUPERVISED)

if __name__ == "__main__":
    semisup()