

import ui
import classifier_config
import dataset_config
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class AppManager:
    def __init__(self):
        self.user_interface = ui.UserInterface()
        self._setup_classification()

    def _load_dataset(self):
        ds = None
        if self.user_interface.dataset == dataset_config.DataseNames.IRIS.value:
            ds = datasets.load_iris()
        elif self.user_interface.dataset == dataset_config.DataseNames.BREAST_CANCER.value:
            ds = datasets.load_breast_cancer()
        elif self.user_interface.dataset == dataset_config.DataseNames.WINE.value:
            ds = datasets.load_wine()
        return ds

    def _load_classifier(self):
        cls = None
        print("self.user_interface.classifier ", self.user_interface.classifier )
        print("self.user_interface.classifier ", classifier_config.ClassifierNames.KNN.value)
        if self.user_interface.classifier == classifier_config.ClassifierNames.KNN.value:
            self.user_interface.add_knn_slider()
            cls = KNeighborsClassifier(n_neighbors=self.user_interface.knn_k)
        elif self.user_interface.classifier == classifier_config.ClassifierNames.SVM.value:
            self.user_interface.add_svm_slider()
            cls = SVC(C = self.user_interface.svm_c)
        elif self.user_interface.classifier == classifier_config.ClassifierNames.RF.value:
            self.user_interface.add_rf_slider()
            cls = RandomForestClassifier(n_estimators= self.user_interface.rf_n_estimators,
                                          max_depth= self.user_interface.rf_max_depth)
        return cls

    def _train_classifier(self, classifier, dataset):
        X = dataset.data
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.user_interface.write(f'Accuracy: {acc}')
        self.user_interface.plot_dataset(dataset)


    def _setup_classification(self):
            print(self.user_interface.dataset)
            print(self.user_interface.classifier)
            ds = self._load_dataset()
            cls = self._load_classifier()
            self._train_classifier(cls, ds)



