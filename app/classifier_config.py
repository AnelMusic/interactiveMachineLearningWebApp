from enum import Enum, auto


class ClassifierNames(Enum):
    KNN = "Key Nearest Neighbour"
    SVM = "Support Vector Machine"
    RF = "Random Forest"
    NB = "Naive Bayes"
    LR = "Logistic Regression"
    DT = "Decision Tree"
