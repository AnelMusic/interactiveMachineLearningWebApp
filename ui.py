import streamlit as st
import classifier_config
import dataset_config

class UserInterface:

    def __init__(self):
        self.classifier_names = classifier_config.ClassifierNames
        self.dataset_names = dataset_config.DataseNames

        self._configure_ui()

    def _configure_ui(self):
        st.title("Interactive Machine Learning Webapp")

        self.selected_classifier = st.sidebar.selectbox("Pick a Classifier", [x.value for x in self.classifier_names])
        self.selected_dataset = st.sidebar.selectbox("Pick a Dataset", [x.value for x in self.dataset_names])

    def write(self, text):
        st.write(text)

    def add_knn_slider(self):
        self.knn_params_k = st.sidebar.slider("k", 1, 15)

    def add_svm_slider(self):
        self.svm_params_c = st.sidebar.slider("C", 0.01, 10.0)

    def add_rf_slider(self):
        self.rf_params_n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        self.rf_params_max_depth = st.sidebar.slider("max_depth", 2, 15)


    @property
    def dataset(self):
        return self.selected_dataset

    @property
    def classifier(self):
        return self.selected_classifier

    @property
    def knn_k(self):
        return self.knn_params_k

    @property
    def svm_c(self):
        return self.svm_params_c

    @property
    def rf_n_estimators(self):
        return self.rf_params_n_estimators

    @property
    def rf_max_depth(self):
        return self.rf_params_max_depth


