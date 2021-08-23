import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


# define the class encodings and reverse encodings
classes = {0: "Bad Risk", 1: "Good Risk"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def init_model():
    clf = GaussianNB()
    df = pd.read_csv("SouthGermanCredit/SouthGermanCredit.asc",skiprows=4,encoding="gbk",engine='python',sep=' ',delimiter=None, index_col=False,header=None,skipinitialspace=True)
    last_ix = len(df.columns) - 1
    X, y = df.drop(last_ix, axis=1), df[last_ix]
    # do the test-train split and train the model
    #Splitting the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2) #80% and 20%
    clf.fit(X_train, y_train)


# function to train and save the model as part of the feedback loop
def train_model(data):
    # load the model
    clf = GaussianNB()

    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.Cost_Matrix_Risk] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)

    # save the model
    #pickle.dump(clf, open("models/iris_nb.pkl", "wb"))
