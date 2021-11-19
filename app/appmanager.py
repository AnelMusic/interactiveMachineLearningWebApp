import classifier_config
import dataset_config
from sklearn import datasets, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import ui


class AppManager:
    def run_app(self):
        self.user_interface = ui.UserInterface()
        accuracy, ds = self._setup_classification()
        self._update_ui(accuracy, ds)

    def _load_dataset(self):
        ds = None
        if self.user_interface.dataset == dataset_config.DataseNames.IRIS.value:
            ds = datasets.load_iris()
        elif (
            self.user_interface.dataset
            == dataset_config.DataseNames.BREAST_CANCER.value
        ):
            ds = datasets.load_breast_cancer()
        elif self.user_interface.dataset == dataset_config.DataseNames.WINE.value:
            ds = datasets.load_wine()
        elif self.user_interface.dataset == dataset_config.DataseNames.DIABETES.value:
            ds = datasets.load_diabetes()
        return ds

    def _load_classifier(self):
        cls = None
        print("self.user_interface.classifier ", self.user_interface.classifier)
        print(
            "self.user_interface.classifier ",
            classifier_config.ClassifierNames.KNN.value,
        )
        if (
            self.user_interface.classifier
            == classifier_config.ClassifierNames.KNN.value
        ):
            self.user_interface.add_knn_slider()
            cls = KNeighborsClassifier(n_neighbors=self.user_interface.knn_k)
        elif (
            self.user_interface.classifier
            == classifier_config.ClassifierNames.SVM.value
        ):
            self.user_interface.add_svm_slider()
            cls = SVC(C=self.user_interface.svm_c)
        elif (
            self.user_interface.classifier == classifier_config.ClassifierNames.RF.value
        ):
            self.user_interface.add_rf_slider()
            cls = RandomForestClassifier(
                n_estimators=self.user_interface.rf_n_estimators,
                max_depth=self.user_interface.rf_max_depth,
            )
        elif (
            self.user_interface.classifier == classifier_config.ClassifierNames.NB.value
        ):
            cls = GaussianNB()
        elif (
            self.user_interface.classifier == classifier_config.ClassifierNames.DT.value
        ):
            cls = tree.DecisionTreeClassifier(random_state=1)
        elif (
            self.user_interface.classifier == classifier_config.ClassifierNames.LR.value
        ):
            self.user_interface.add_lr_slider()
            cls = LogisticRegression(max_iter=self.user_interface.lr_max_iter)
        return cls

    def _train_classifier(self, classifier, dataset):
        X = dataset.data
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=12345
        )
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc

    def _plot_dataset(self, dataset, label_x, label_y):
        self.user_interface.plot_dataset(dataset, label_x, label_y)

    def _update_ui(self, accuracy, dataset):
        self.user_interface.write_sidebar("### Accuracy:")
        self.user_interface.write_sidebar(f"{accuracy}")
        self._plot_dataset(dataset, "Principal Component 1", "Principal Component 2")
        self.user_interface.write_sidebar("### Dataset Shape:")
        self.user_interface.write_sidebar(f"{dataset.data.shape}")

    def _setup_classification(self):
        print(self.user_interface.dataset)
        print(self.user_interface.classifier)
        ds = self._load_dataset()
        cls = self._load_classifier()
        acc = self._train_classifier(cls, ds)
        return acc, ds
