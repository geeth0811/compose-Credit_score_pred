import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# define a Gaussain NB classifier
clf = GaussianNB()

# define the class encodings and reverse encodings
classes = {0: "Bad Risk", 1: "Good Risk"}
r_classes = {y: x for x, y in classes.items()}


# function to load the model
def load_model():
    global clf
    df = pd.read_csv("SouthGermanCredit/SouthGermanCredit.asc",skiprows=4,encoding="gbk",engine='python',sep=' ',delimiter=None, index_col=False,header=None,skipinitialspace=True)
    last_ix = len(df.columns) - 1
    X, y = df.drop(last_ix, axis=1), df[last_ix]
    # do the test-train split and train the model
    #Splitting the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2) #80% and 20%
    clf.fit(X_train, y_train)


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    return classes[prediction]
