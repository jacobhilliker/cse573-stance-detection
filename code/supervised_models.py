"""
Author: Jack Myers and Henry Zhao
Purpose: Train and evaluate four different supervised classification models
"""

from sklearn import svm, neighbors, naive_bayes
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import fasttext
from process_data import *
import json

def train_with_data():
    train_data = load_corrected_data("data/semeval2016_corrected_train.csv")
    test_data = load_corrected_data("data/semeval2016_corrected_test.csv")

    return train_models(train_data + test_data)

def train_models(data):
    """
    Load train and testing data, and then train all four models
    """
    # # uncomment this method to split the data into train and test csv files
    # split_data()

    # load train and test datasets
    len_train_data = len(load_corrected_data("data/semeval2016_corrected_train.csv"))

    # combine the sets to vectorize them, and then split them again after vectors computed
    vectorized_data = vectorize(data)

    # split vectorized data back to train and test datasets
    train_data = vectorized_data[: len_train_data]
    test_data = vectorized_data[len_train_data :]

    # format training data into required file for fastText model
    with open("data/fasttext_train.txt", "w") as file:
        for tweet in train_data:
            try:
                file.write(f"{tweet['tweet']} __label__{tweet['stance']}\n")
            except:
                # ignore invalid input rows for training
                pass

    # train the fastText model, it only takes the X_test and y_test as input
    fastText = train_fastText(get_tweets(test_data), get_stances(test_data))

    # Train the three sklearn models. Each takes X_train, X_test, y_train, y_test as input
    knn = train_knn(get_vectors(train_data), get_vectors(test_data), get_stances(train_data), get_stances(test_data))
    naive_bayes = train_naive_bayes(get_vectors(train_data), get_vectors(test_data), get_stances(train_data), get_stances(test_data))
    svm = train_svm(get_vectors(train_data), get_vectors(test_data), get_stances(train_data), get_stances(test_data)) 

    results = json.dumps([fastText, knn, naive_bayes, svm])

    return results


def train_fastText(x_test, y_test):
    """
    train the fastText model, and then evaluate it using the testing data
    """
    y_pred = []

    # Try to load a previous model, if it fails then train a new one and save it.
    # If you wish to retrain the model, then delete the bin file before executing this method
    try:
        model = fasttext.load_model("models/fasttext_trained_model.bin")
    except:
        model = fasttext.train_supervised(
            "data/fasttext_train.txt", lr=0.006, dim=20, epoch=500
        )
        # model = fasttext.train_supervised('data/fasttext_train.txt', autotuneValidationFile='data/val.txt', autotuneDuration=600)
        model.save_model("models/fasttext_trained_model.bin")

    # get all predicted labels and compare with test labels
    for x in x_test:
        pred = model.predict(x)
        y_pred.append(pred[0][0].replace("__label__", ""))

    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    return json.dumps({
        'model': 'fastText',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1})


'''
train the knn model, and then evaluate it
'''
def train_knn(x_train, x_test, y_train, y_test):
    """
    train the knn model, and then evaluate it
    """
    k = 13
    knn = neighbors.KNeighborsClassifier(k, weights="distance")
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    return json.dumps({
        'model': 'knn',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1})


'''
train the Gaussian naive bayes model, and evaluate it
'''
def train_naive_bayes(x_train, x_test, y_train, y_test):
    """
    train the Gaussian naive bayes model, and evaluate it
    """
    nb = naive_bayes.ComplementNB(alpha=2)
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    return json.dumps({
        'model': 'naive_bayes',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1})

'''
train the linear svc model, and evaluate it
'''
def train_svm(x_train, x_test, y_train, y_test):
    """
    train the linear svc model, and evaluate it
    """
    svc = svm.LinearSVC(C=1.5, penalty="l1", loss="squared_hinge", dual=False)
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    return json.dumps({
        'model': 'svm',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1})


if __name__ == '__main__':
    train_with_data()