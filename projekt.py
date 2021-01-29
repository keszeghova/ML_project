import numpy as np
np.random.seed(1337)
import tensorflow as tf
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical


def load_data(path):
    data_X = []
    data_Y = []
    for fn in os.listdir(path):
        fpath = os.path.join(path, fn)
        for file in os.listdir(fpath):
            fpath2 = os.path.join(fpath, file)
            ds = pydicom.read_file(fpath2)
            n = len(ds.pixel_array)*len(ds.pixel_array[0])
            data_X.append(np.array(ds.pixel_array).reshape(n))
            if 'true' in fpath2:
                data_Y.append(1)
            else:
                data_Y.append(0)
                
    return data_X, data_Y

def load_data_2D(path):
    data_X = []
    data_Y = []
    for fn in os.listdir(path):
        fpath = os.path.join(path, fn)
        for file in os.listdir(fpath):
            fpath2 = os.path.join(fpath, file)
            ds = pydicom.read_file(fpath2)
            data_X.append(np.array(ds.pixel_array))
            if 'true' in fpath2:
                data_Y.append(1)
            else:
                data_Y.append(0)

    return data_X, data_Y


def neural_net(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12)

    n = 512
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = X_train.reshape(-1, n, n, 1)
    X_test = X_test.reshape(-1, n, n, 1)
    input_shape = (n, n, 1)

    print("New train data shape: {} | test data: {}".format(X_train.shape, X_test.shape))
    
    cnn = Sequential()
    cnn.add(Conv2D(16, (11, 11), activation='elu', input_shape=input_shape))
    cnn.add(MaxPooling2D(pool_size=(4, 4)))
    
    cnn.add(Conv2D(32, (5, 5), activation='elu'))
    cnn.add(MaxPooling2D(pool_size=(4, 4)))
    
    cnn.add(Conv2D(64, (3, 3), activation='elu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Conv2D(128, (3, 3), activation='elu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Flatten())

    cnn.add(Dense(10, activation='relu'))

    n_classes = 2
    cnn.add(Dense(n_classes, activation='softmax'))

    cnn.compile(loss='categorical_crossentropy', optimizer=Adam(0.001),  metrics=['accuracy'])
    #cnn.compile(loss='categorical_crossentropy', optimizer=SGD(0.001),  metrics=['accuracy'])

    history = cnn.fit(X_train, to_categorical(y_train), epochs=5, batch_size=40, 
                      validation_split=0.1, verbose=True)


    train_score = cnn.evaluate(X_train, to_categorical(y_train))
    print("\n\ntrain loss: {} | train acc: {}\n".format(train_score[0], train_score[1]))

    test_score = cnn.evaluate(X_test, to_categorical(y_test))
    print("\n\ntest loss: {} | test acc: {}".format(test_score[0], test_score[1]))
    

def train_PCA(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    pca = PCA(n_components=0.99)
    f = pca.fit(X_train)
    ev = f.explained_variance_
    evr = f.explained_variance_ratio_
    x = sum(evr)*100
    print(f'variancia: {x}%')

    reduced_X_train, reduced_X_test = pca.transform(X_train), pca.transform(X_test)
    clf = svm.SVC(kernel='rbf', C=1).fit(reduced_X_train, y_train)
    #clf = AdaBoostClassifier(n_estimators=150, random_state=0)
    #clf.fit(reduced_X_train, y_train)
    print(clf.score(reduced_X_test, y_test))

    '''
    -> svm.SVC
    variancia: 99.00900713620035%
    0.8307692307692308

    ->AdaBoostClassifier
    variancia: 99.01278887191943%
    0.8307692307692308
    '''

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

##    for c in range(1, 6):
##        for k in ['linear', 'poly', 'rbf', 'sigmoid']:
##            clf = svm.SVC(kernel=k, C=c).fit(X_train, y_train)
##            print(c, k, clf.score(X_test, y_test))
    
    clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
    #0.8769230769230769

    #clf =  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(X_train, y_train)
    #0.7461538461538462

    #clf = KNeighborsClassifier(2).fit(X_train, y_train)
    #0.9076923076923077
    
    print(clf.score(X_test, y_test))
    

path = 'C:\\Users\\daska\\Documents\\AIN\\MAGISTER\\Diplomovka\\Data'
#X, Y = load_data(path)
#train(X, Y)
#train_PCA(X, Y)

X, Y = load_data_2D(path)
neural_net(X, Y)


