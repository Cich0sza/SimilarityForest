import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from DataParser import DataParser

from simforest import SimilarityForest


if __name__ == '__main__':
    y, X = DataParser("data\\a1a.txt", 123).parse()
    # y, X = DataParser("data\\breast-cancer.txt", 10).parse()
    # y, X = DataParser("data\\german-numer.txt", 24).parse()
    # y, X = DataParser("data\\heart.txt", 13).parse()
    # y, X = DataParser("data\\ionosphere_scale.txt", 34).parse()
    # y, X = DataParser("data\\mushrooms.txt", 112).parse()
    # y, X = DataParser("data\\splice.txt", 60).parse()

    x = X.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X = pd.DataFrame(x_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    sim_forest = SimilarityForest(n_estimators=20)
    sim_forest.fit(X_train, y_train)
    sf_pred = sim_forest.predict(X_test)
    # sf_pred = sim_forest.predict_probability(X_test)
    print(sf_pred)
    print("Similarity forest")
    print(accuracy_score(y_test, sf_pred))
    print(confusion_matrix(y_test, sf_pred))

    rf = RandomForestClassifier(n_estimators=20)
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    print("Random forest")
    print(accuracy_score(y_test, rf_pred))
    print(confusion_matrix(y_test, rf_pred))
