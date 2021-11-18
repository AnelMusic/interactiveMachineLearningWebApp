import streamlit as st
from enum import Enum, auto

class ClassifierNames(Enum):
    KNN = "Key Nearest Neighbour"
    SVM = "Support Vector Machine"
    RF = "Random Forest"

class DataseNames(Enum):
    IRIS = "Iris Dataset"
    WINE = "Wine Dataset"
    BREAST_CANCER = "Breast Cancer Dataset"

class UserInterface:
    classifier_names : ClassifierNames
    dataset_names : DataseNames


    def __init__(self):
        self.classifier_names = ClassifierNames
        self.dataset_names = DataseNames

        self._configure_ui()

    def _configure_ui(self):
        st.title("Interactive Machine Learning Webapp")
        st.write("# Pick a classifier and see it's performance")

        self.classifier_names_sbox = st.sidebar.selectbox("Pick a Classifier", [x.value for x in self.classifier_names])
        self.dataset_names_sbox = st.sidebar.selectbox("Pick a Dataset", [x.value for x in self.dataset_names])

    @property
    def selected_dataset(self):
        return self.dataset_names_sbox



