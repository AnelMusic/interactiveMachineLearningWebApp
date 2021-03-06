import classifier_config
import dataset_config
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


class UserInterface:
    def __init__(self):
        self.classifier_names = classifier_config.ClassifierNames
        self.dataset_names = dataset_config.DataseNames
        self._configure_ui()

    def _configure_ui(self):
        st.title("Interactive Machine Learning Webapp")
        self.selected_classifier = st.sidebar.selectbox(
            "Pick a Classifier", [x.value for x in self.classifier_names]
        )
        self.selected_dataset = st.sidebar.selectbox(
            "Pick a Dataset", [x.value for x in self.dataset_names]
        )

    def write(self, text):
        st.write(text)

    def write_sidebar(self, text):
        st.sidebar.write(text)

    def _plot_dataset_2D_PCA(self, ds, x_label, y_label):
        y = ds.target
        fig1 = plt.figure(1, figsize=(8, 6))
        X_reduced = PCA(n_components=2).fit_transform(ds.data)
        plt.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=y,
            cmap="viridis",
            s=40,
        )
        plt.title("PCA")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.colorbar()
        st.pyplot(fig1)

    def _plot_dataset_3D_PCA(self, ds):
        # To getter a better understanding of interaction of the dimensions
        # plot the first three PCA dimensions
        y = ds.target
        fig2 = plt.figure(2, figsize=(8, 6))
        ax = Axes3D(fig2, elev=-160, azim=120)
        X_reduced = PCA(n_components=3).fit_transform(ds.data)
        ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            X_reduced[:, 2],
            c=y,
            cmap="viridis",
            edgecolor="k",
            s=40,
        )
        ax.set_title("First three PCA directions")
        ax.set_xlabel("1st eigenvector")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd eigenvector")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd eigenvector")
        ax.w_zaxis.set_ticklabels([])

        st.pyplot(fig2)

    def plot_dataset(self, ds, x_label, y_label):
        self._plot_dataset_2D_PCA(ds, x_label, y_label)
        self._plot_dataset_3D_PCA(ds)

    def add_knn_slider(self):
        self.knn_params_k = st.sidebar.slider("k", 1, 15)

    def add_svm_slider(self):
        self.svm_params_c = st.sidebar.slider("C", 0.01, 10.0)

    def add_rf_slider(self):
        self.rf_params_n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        self.rf_params_max_depth = st.sidebar.slider("max_depth", 2, 15)

    def add_lr_slider(self):
        self.lr_params_max_iter = st.sidebar.slider("max_iter", 1, 2000)

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

    @property
    def lr_max_iter(self):
        return self.lr_params_max_iter
