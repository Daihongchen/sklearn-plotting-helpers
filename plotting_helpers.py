from itertools import product

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_roc_curve_custom(classifier, X, y, name, ax, **kwargs):
    """Plot Receiver operating characteristic (ROC) curve.
    Extra keyword arguments will be passed to matplotlib's `plot`.
    Parameters
    ----------
    classifier : estimator instance
        Trained classifier.
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.
    y : array-like of shape (n_samples,)
        Target values.
    name : str
        Name of ROC Curve for labeling.
    ax : matplotlib axes
        Axes object to plot on.
    Returns
    -------
    None
    """
    # calculate ROC curve and AUC
    y_pred = classifier.predict_proba(X)[:, 1]
    false_positive, true_positive, _ = roc_curve(y, y_pred, drop_intermediate=False)
    roc_auc = auc(false_positive, true_positive)

    # plot curves as a labeled line graph
    ax.plot(false_positive, true_positive,
            label="{} (AUC = {:0.2f})".format(name, roc_auc), **kwargs)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc='lower right')


def plot_confusion_matrix_custom(classifier, X, y, name, ax, normalize=None, cmap="viridis"):
    """Plot Confusion Matrix.
    Parameters
    ----------
    classifier : estimator instance
        Trained classifier.
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.
    y : array-like of shape (n_samples,)
        Target values.
    name : str
        Name of model for labeling.
    ax : matplotlib Axes
        Axes object to plot on.
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    cmap : str or matplotlib Colormap, default='viridis'
        Colormap recognized by matplotlib.
    Returns
    -------
    None
    """
    # calculate confusion matrix
    y_pred = classifier.predict(X)
    cm = confusion_matrix(y, y_pred, normalize=normalize)

    # create an image
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)

    # print text with appropriate color depending on background
    text = np.empty_like(cm, dtype=object)
    n_classes = cm.shape[0]
    display_labels = classifier.classes_
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        text[i, j] = ax.text(j, i, cm[i, j], ha="center",
                             va="center", color=color)

    # attach color bar for legend
    fig = ax.figure
    fig.colorbar(im, ax=ax, shrink=0.67)

    # customize appearance
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels,
           ylabel="True label",
           xlabel="Predicted label",
           title=name)
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation="horizontal")

