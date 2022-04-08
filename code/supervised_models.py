from sklearn import svm, neighbors, naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import fasttext
from process_data import *



def train_models():
    # # uncomment this method to split the corpus into train and test csv files
    # split_data()

    # load train and test datasets
    train_data = load_corrected_data('data/semeval2016_corrected_train.csv')
    test_data = load_corrected_data('data/semeval2016_corrected_test.csv')

    # combine the sets to vectorize them, and then split them again after vectors computed
    vectorized_data = vectorize(train_data + test_data)

    # split vectorized data back to train and test datasets
    train_data = vectorized_data[:len(train_data)]
    test_data = vectorized_data[len(train_data):]    
    
    # Train the three sklearn models. Each takes X_train, X_test, y_train, y_test as input

    train_knn(get_vectors(train_data), get_vectors(test_data), get_stances(train_data), get_stances(test_data))
    train_naive_bayes(get_vectors(train_data), get_vectors(test_data), get_stances(train_data), get_stances(test_data))
    train_svm(get_vectors(train_data), get_vectors(test_data), get_stances(train_data), get_stances(test_data)) 

    # format training data into required file for fastText model
    with open("data/fasttext_train.txt", 'w') as file:
        for tweet in train_data:
            try:
                file.write("{} __label__{}\n".format(tweet['tweet'], tweet['stance']))
            except:
                # ignore invalid input rows for training
                pass
    file.close()

    train_fastText(get_tweets(test_data), get_stances(test_data))


def train_fastText(x_test, y_test):
    y_pred = []

    # try to load a previous model, if it fails then train a new one and save it
    try:
        model = fasttext.load_model('models/fasttext_trained_model.bin')
    except:
        model = fasttext.train_supervised('data/fasttext_train.txt', lr=0.001, dim=500, epoch=5000)
        model.save_model('models/fasttext_trained_model.bin')
    for x in x_test:
        pred = model.predict(x)
        y_pred.append(pred[0][0].replace('__label__',''))
        
    
    accuracy = accuracy_score(y_test, y_pred)
    print('fastText', accuracy)


def train_knn(x_train, x_test, y_train, y_test):
    k = 15
    knn = neighbors.KNeighborsClassifier(k, weights="uniform") # uniform or distance
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('KNN', accuracy)


def train_naive_bayes(x_train, x_test, y_train, y_test):
    nb = naive_bayes.GaussianNB()
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Naive Bayes', accuracy)


def train_svm(x_train, x_test, y_train, y_test):
    svc = svm.LinearSVC()
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('SVM', accuracy)

if __name__ == '__main__':
    train_models()