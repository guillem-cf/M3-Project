import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import normalize,  LabelBinarizer, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture as GMM

import matplotlib.pyplot as plt
from functools import partial
import time
import pandas as pd
from tqdm.notebook import trange, tqdm
import optuna
from optuna.visualization.matplotlib import plot_contour, plot_edf, plot_intermediate_values, plot_optimization_history, plot_parallel_coordinate, plot_param_importances, plot_slice, plot_pareto_front
import os
from optuna.samplers import TPESampler
import concurrent.futures
import gc
import seaborn as sns
from itertools import cycle
import random

# Function that normalizes an array x using L^2 norm
def l2_norm(x):
    norm = np.linalg.norm(x, ord=2)
    x_norm = x / norm
    return x_norm
# Function that normalizes an array x using power norm
def power_norm(x):
    x = np.sign(x) * np.abs(x) ** 0.5
    norm = np.linalg.norm(x, ord=2)
    x_norm = x / norm
    return x_norm
def no_norm(x):
    return x

class Dummy:
    """Dummy dimensionality reduction method that keeps all the original features."""

    def fit_transform(self, features, labels):
        return features

    def transform(self, features):
        return features


norms = {"l2": l2_norm, "power": power_norm, "none": no_norm}

classifiers = {"KNN": KNeighborsClassifier, "svm": SVC}

dim_reduction = {
    "None": Dummy,
    "PCA": PCA,
    "LDA": LinearDiscriminantAnalysis,
}

kernels = {
    "linear": "linear",
    "RBF": "rbf",
    # "histogram_intersection": histogram_intersection,
}

metrics = {
    "balanced_accuracy": balanced_accuracy_score,
    "accuracy": accuracy_score,
    "f1-score": f1_score,
    "confusion-matrix": confusion_matrix,
}



def cluster_local_features(n_clusters,features):
    codebook = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init="auto",
        verbose=False,
        batch_size=min(20 * n_clusters, features.shape[0]),
        compute_labels=False,
        reassignment_ratio=10**-4,
        random_state=42
    )
    codebook.fit(features)
    return codebook




def compute_histogram(assigned_clusters, num_clusters, normalization=norms["l2"]):
    bag_visual_words = np.zeros((len(assigned_clusters), num_clusters), dtype=np.float32)

    for i in range(len(assigned_clusters)):
        hist_i, _ = np.histogram(assigned_clusters[i], bins=num_clusters, range=(0, num_clusters))
        bag_visual_words[i, :] = normalization(hist_i)

    return bag_visual_words



def obtain_spatial_histogram_visual_words(
    features,
    position_features,
    tr_lengths=None,
    codebook=None,
    normalization=norms["l2"],
    pyramid=None,
):

    if tr_lengths is None:
        tr_lengths = [len(feature) for feature in features]
        features = np.vstack(features)

    assigned_labels = codebook.predict(features)

    lengths = np.array([0] + [descriptor_length for descriptor_length in tr_lengths])
    lengths = np.cumsum(lengths)

    splitted_labels = [
        assigned_labels[lengths[i] : lengths[i + 1]] for i in range(len(lengths) - 1)
    ]

    if pyramid is None:
        return compute_histogram(
            splitted_labels,
            codebook.cluster_centers_.shape[0],
            normalization=normalization,
        )

    num_clusters = codebook.cluster_centers_.shape[0]
    histograms = np.zeros((len(splitted_labels), num_clusters * pyramid["size"]), dtype=np.float32)

    def get_positions(position_features, x0, y0, x, y):
        return np.where(
            (position_features[:, 0] >= x0)
            & (position_features[:, 1] >= y0)
            & (position_features[:, 0] < x)
            & (position_features[:, 1] < y)
        )

    num_patches = 0
    for (x0, y0, x, y) in list(pyramid["iterate"]()):
        # Get the labels of the features that are in the current division
        splitted_labels = [
            assigned_labels[lengths[i] : lengths[i + 1]][
                get_positions(position_features[lengths[i] : lengths[i + 1]], x0, y0, x, y)
            ]
            for i in range(len(lengths) - 1)
        ]
        # Compute the histogram of the current division
        histograms[
            :, num_patches * num_clusters : (num_patches + 1) * num_clusters
        ] = compute_histogram(splitted_labels, num_clusters, normalization=normalization)
        num_patches += 1

    return histograms


class BoVWClassifier(BaseEstimator, ClassifierMixin):
    """Image classifier using Bag of Visual Words."""

    def __init__(
        self,
        clustering_method,
        classifier,
        reduction_method,
        normalization,
        spatial_pyramid_div=None,
    ):
        self.clustering_method = clustering_method
        self.classifier = classifier
        self.reduction_method = reduction_method
        self.codebook = None
        self.normalization = normalization
        self.spatial_pyramid_div = spatial_pyramid_div

    def fit(self, features, labels, sample_weight=None):
        tr_lengths = [len(feature) for feature in features]
        features = np.vstack(features)
        position_features, features = features[:, :2], features[:, 2:]
        self.codebook = self.clustering_method(features)

        tr_hist = obtain_spatial_histogram_visual_words(
            features,
            position_features,
            tr_lengths,
            self.codebook,
            self.normalization,
            self.spatial_pyramid_div,
        )

        tr_hist_reduced = self.reduction_method.fit_transform(tr_hist, labels)

        # Standardize features by removing the mean and scaling to unit variance.
        self.scaler = StandardScaler()
        tr_hist_reduced = self.scaler.fit_transform(tr_hist_reduced)

        self.classifier.fit(tr_hist_reduced, labels)

    def fit_transform(self, features, labels):
        self.fit(features, labels)
        return self.predict(features)

    def predict_proba(self, features):
        te_lengths = [len(feature) for feature in features]
        features = np.vstack(features)
        position_features, features = features[:, :2], features[:, 2:]

        te_hist = obtain_spatial_histogram_visual_words(
            features,
            position_features,
            te_lengths,
            self.codebook,
            self.normalization,
            self.spatial_pyramid_div,
        )
        te_hist_reduced = self.reduction_method.transform(te_hist)
        te_hist_reduced = self.scaler.transform(te_hist_reduced)
        cls = self.classifier.predict_proba(te_hist_reduced)
        return cls

    def predict(self, features):
        te_lengths = [len(feature) for feature in features]
        features = np.vstack(features)
        position_features, features = features[:, :2], features[:, 2:]

        te_hist = obtain_spatial_histogram_visual_words(
            features,
            position_features,
            te_lengths,
            self.codebook,
            self.normalization,
            self.spatial_pyramid_div,
        )
        te_hist_reduced = self.reduction_method.transform(te_hist)
        te_hist_reduced = self.scaler.transform(te_hist_reduced)
        cls = self.classifier.predict(te_hist_reduced)
        return cls

    def score(self, X, y=None):
        return sum(self.predict(X))

    def score_accuracy(self, X, y):
        return 100 * self.score(X, y) / len(y)


# Define a function to compute the f1-score and accuracy for each class
def compute_metrics(truth, preds):
    results = []
    unique_labels = np.unique(truth)
    truth, preds = np.array(truth), np.array(preds)
    for lab in unique_labels:
        acc = metrics["accuracy"](truth == lab, preds == lab)
        F1 = metrics["f1-score"](truth == lab, preds == lab)
        results.append((lab, acc, F1))

    overall_acc = metrics["accuracy"](truth, preds)
    overall_bal_acc = metrics["balanced_accuracy"](truth, preds)
    weighted_F1 = metrics["f1-score"](truth, preds, average="weighted")
    results.append(("OVERALL (acc)", overall_acc, weighted_F1))
    results.append(("OVERALL (balanced acc)", overall_bal_acc, weighted_F1))
    return pd.DataFrame(data=results, columns=["label", "accuracy", "f1_score"])


# Apply the Bag of Visual Words model to the features
def BoVW(train_descriptors, train_labels_descrip, test_descriptors, test_labels_descrip, 
            n_clusters = 798, 
            dim_red = "None", n_components = 69, 
            kernel_type = "RBF", best_gamma= 0.004454186007581258, best_C = 4.380442487942557, probability = True, 
            best_norm = "power"):
    
    clustering = partial(cluster_local_features, n_clusters)
    # reduction_method = dim_reduction[dim_red](n_components)
    reduction_method = dim_reduction[dim_red]()
    classifier = classifiers["svm"](
        kernel=kernels[kernel_type], C=best_C, gamma=best_gamma, probability=probability)
    normalization = norms[best_norm]

    ex_trainer = BoVWClassifier(clustering, classifier, reduction_method, normalization, spatial_pyramid_div=None)
    ex_trainer.fit(train_descriptors, train_labels_descrip)

    predictions = ex_trainer.predict(test_descriptors)

    # accuracy = ex_trainer.score(test_descriptors, test_labels_descrip)

    scores = compute_metrics(test_labels_descrip, predictions)

    return scores


def svm(train_descriptors, train_labels_descrip, test_descriptors, test_labels_descrip, 
            kernel_type = "RBF", best_gamma= 0.004454186007581258, best_C = 4.380442487942557, probability = True):
    
    scaler = StandardScaler()
    scaler.fit(train_descriptors)
    train_descriptors = scaler.transform(train_descriptors)
    test_descriptors = scaler.transform(test_descriptors)

    classifier = classifiers["svm"](
        kernel=kernels[kernel_type], C=best_C, gamma=best_gamma, probability=probability)
    classifier.fit(train_descriptors, train_labels_descrip)

    predictions = classifier.predict(test_descriptors)

    accuracy = classifier.score(test_descriptors, test_labels_descrip)

    scores = compute_metrics(test_labels_descrip, predictions)

    return classifier, scores, accuracy



def plotROC_BWVW(
    train_labels_descrip,
    test_labels_descrip,
    train_descriptors,
    test_descriptors,
    labels,
    classifier,
):

    y_onehot_train = LabelBinarizer().fit_transform(train_labels_descrip)
    y_onehot_test = LabelBinarizer().fit_transform(test_labels_descrip)
    n_samples, n_classes = y_onehot_test.shape

    clf = OneVsRestClassifier(classifier, n_jobs=8)
    clf.fit(train_descriptors, y_onehot_train)
    y_score = clf.predict_proba(test_descriptors)

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

    fig, ax = plt.subplots(figsize=(8, 8))

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(
        [
            "aqua",
            "darkorange",
            "cornflowerblue",
            "red",
            "green",
            "magenta",
            "blue",
            "pink",
        ]
    )
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {labels[class_id]}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest multiclass ROC")
    plt.legend()
    plt.show()


