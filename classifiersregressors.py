from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

def RunMLR(x_train,y_train,x_test,y_test):
    mlr = LinearRegression()
    mlr.fit(x_train,y_train)
    #y_predict = mlr.predict(x_test)
    print("MLR: " + str(mlr.score(x_test,y_test)))

def RunKNNRegression(x_train,y_train,x_test,y_test):
    kreg = KNeighborsRegressor(n_neighbors=14)
    kreg.fit(x_train,y_train)
    #y_predict = kreg.predict(x_test)
    print("KNNReg: " + str(kreg.score(x_test,y_test)))

def ChartKNNRegressor(x_train,y_train,x_test,y_test):
    #iterate K and plot
    kguesses = []
    ks = []
    y_train_flat = y_train.values.ravel()
    for i in range(1,50):
        ks.append(i)
        classifier = KNeighborsRegressor(n_neighbors = i)
        classifier.fit(x_train,y_train_flat)
        kguesses.append(classifier.score(x_test,y_test))

    plt.plot(ks,kguesses,alpha=.3)
    plt.show()


def ChartKNNClassifier(x_train,y_train,x_test,y_test):
    #iterate K and plot
    kguesses = []
    ks = []
    y_train_flat = y_train.values.ravel()
    for i in range(1,50):
        ks.append(i)
        classifier = KNeighborsClassifier(n_neighbors = i)
        classifier.fit(x_train,y_train_flat)
        kguesses.append(classifier.score(x_test,y_test))

    plt.plot(ks,kguesses,alpha=.3)
    plt.show()

def RunKNNClassifier(x_train,y_train,x_test,y_test):
    y_train_flat = y_train.values.ravel()
    classifier = KNeighborsClassifier(n_neighbors = 44)
    classifier.fit(x_train,y_train_flat)
    KAgeGuess = classifier.predict(x_test)
    print("K Score: " + str(classifier.score(x_test,y_test)))
    print(accuracy_score(y_test,KAgeGuess))
    print(recall_score(y_test,KAgeGuess,average='micro'))
    print(precision_score(y_test,KAgeGuess,average='micro'))
    print(f1_score(y_test,KAgeGuess,average='micro'))

# #svc classifier
def RunSVC(x_train,y_train,x_test,y_test):
    svcclassifier = SVC(gamma=5,C=.7)
    svcclassifier.fit(x_train,y_train)
    svcPredict = svcclassifier.predict(x_test)
    print("svc Score: " + str(svcclassifier.score(x_test,y_test)))
    print(accuracy_score(y_test,svcPredict))
    print(recall_score(y_test,svcPredict,average='micro'))
    print(precision_score(y_test,svcPredict,average='micro'))
    print(f1_score(y_test,svcPredict,average='micro'))
