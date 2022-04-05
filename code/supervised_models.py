from sklearn import svm, neighbors, naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import fasttext
from process_data import *



def train_models():
    data = vectorize(load_corrected_data())

    x = get_vectors(data)
    y = get_stances(data)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    train_knn(x_train, x_test, y_train, y_test)
    train_naive_bayes(x_train, x_test, y_train, y_test)
    train_svm(x_train, x_test, y_train, y_test)


def train_fastText():
    pass

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