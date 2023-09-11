import warnings
from numbers import Integral, Real

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_datasets(transcripts_true, transcripts_lie, train_ratio=0.7, rng=None):
    # check that transcripts_true and transcripts_lie are of the same type
    assert isinstance(transcripts_true, type(transcripts_lie))
    if rng is None:
        rng = np.random.RandomState()
    # create the dataset
    if isinstance(transcripts_true, pd.Series):
        X = pd.concat([transcripts_true, transcripts_lie], ignore_index=True)
    else:
        # concatenate along the index dimension
        X = pd.concat([transcripts_true, transcripts_lie], axis=0)
    y = pd.Series([1] * len(transcripts_true) + [0] * len(transcripts_lie))
    # shuffle:
    indeces = rng.permutation(np.arange(len(X)))
    X, y = X.iloc[indeces], y.iloc[indeces]
    # split into train and test
    X_train, X_test = X.iloc[:int(len(X) * train_ratio)], X.iloc[int(len(X) * train_ratio):]
    y_train, y_test = y.iloc[:int(len(y) * train_ratio)], y.iloc[int(len(y) * train_ratio):]
    # now convert into numpy arrays if they are pd.Series
    if isinstance(X_train, pd.Series):
        X_train = np.array(X_train.tolist())
        X_test = np.array(X_test.tolist())
    y_train = np.array(y_train.tolist())  # this is always a Series
    y_test = np.array(y_test.tolist())  # this is always a Series
    return X_train, X_test, y_train, y_test


def create_datasets_paired_questions(dataset, dataset2=None, train_ratio=0.7, rng=None):
    """`dataset` and `dataset2` are here dataframew where each row corresponds to a question, the first column to the truthful case and the second one to the lying case. """
    assert len(dataset.shape) == 2
    assert "truth" in dataset.columns[0]
    assert "lie" in dataset.columns[1]
    truth_instructions_present = "truth_instruction" in dataset.columns
    lie_instructions_present = "lie_instruction" in dataset.columns
    assert truth_instructions_present == lie_instructions_present
    dataset_present = "dataset" in dataset.columns
    if dataset2 is not None:
        assert len(dataset2.shape) == 2
    if rng is None:
        rng = np.random.RandomState()
    # train-test split:
    # shuffle rows:
    indeces = rng.permutation(np.arange(dataset.shape[0]))
    dataset = dataset.iloc[indeces]
    X_train, X_test = dataset.iloc[:int(len(dataset) * train_ratio)], dataset.iloc[int(len(dataset) * train_ratio):]

    if truth_instructions_present:
        train_truth_instructions = X_train["truth_instruction"]
        train_lie_instructions = X_train["lie_instruction"]
        test_truth_instructions = X_test["truth_instruction"]
        test_lie_instructions = X_test["lie_instruction"]
        train_instructions = pd.concat([train_truth_instructions, train_lie_instructions])
        test_instructions = pd.concat([test_truth_instructions, test_lie_instructions])
    if dataset_present:
        train_dataset = X_train["dataset"]
        test_dataset = X_test["dataset"]
        train_datasets = pd.concat([train_dataset, train_dataset])
        test_datasets = pd.concat([test_dataset, test_dataset])

    # now concatenate the first two columns
    X_train = np.concatenate([np.stack([X_train.iloc[i, 0] for i in range(X_train.shape[0])]),
                              np.stack([X_train.iloc[i, 1] for i in range(X_train.shape[0])])])
    X_test = np.concatenate([np.stack([X_test.iloc[i, 0] for i in range(X_test.shape[0])]),
                             np.stack([X_test.iloc[i, 1] for i in range(X_test.shape[0])])])
    # create y
    y_train = np.concatenate([np.ones(len(X_train) // 2), np.zeros(len(X_train) // 2)])
    y_test = np.concatenate([np.ones(len(X_test) // 2), np.zeros(len(X_test) // 2)])
    if dataset2 is not None:
        # shuffle rows:
        dataset2 = dataset2.iloc[indeces]
        X_train2, X_test2 = dataset2.iloc[:int(len(dataset2) * train_ratio)], dataset2.iloc[
                                                                              int(len(dataset2) * train_ratio):]
        # now concatenate the two columns
        X_train2 = np.concatenate([np.stack([X_train2.iloc[i, 0] for i in range(X_train2.shape[0])]),
                                   np.stack([X_train2.iloc[i, 1] for i in range(X_train2.shape[0])])])
        X_test2 = np.concatenate([np.stack([X_test2.iloc[i, 0] for i in range(X_test2.shape[0])]),
                                  np.stack([X_test2.iloc[i, 1] for i in range(X_test2.shape[0])])])
    # shuffle again:
    indeces_train = rng.permutation(np.arange(len(X_train)))
    X_train, y_train = X_train[indeces_train], y_train[indeces_train]
    indeces_test = rng.permutation(np.arange(len(X_test)))
    X_test, y_test = X_test[indeces_test], y_test[indeces_test]
    return_args = [X_train, X_test]
    if truth_instructions_present:
        train_instructions = train_instructions.iloc[indeces_train]
        test_instructions = test_instructions.iloc[indeces_test]
        return_args += [train_instructions, test_instructions]
    if dataset_present:
        train_datasets = train_datasets.iloc[indeces_train]
        test_datasets = test_datasets.iloc[indeces_test]
        return_args += [train_datasets, test_datasets]
    if dataset2 is not None:
        # shuffle again:
        X_train2 = X_train2[indeces_train]
        X_test2 = X_test2[indeces_test]
        return_args += [X_train2, X_test2]
    return_args += [y_train, y_test]
    return return_args


class Classifier:
    def __init__(self, X_train, y_train, classifier="logistic", scale=True, **kwargs):
        """Fit a classifier model and return the accuracy, AUC and confusion matrix.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Training data.
        y_train : array-like, shape (n_samples,)
            Target values.
        classifier : str, default="logistic"
            The model to use. One of "logistic", "random_forest", "MLP", "SVM", "ada_boost", "gradient_boosting".
        scale : bool, default=True
            Whether to scale the data.
        max_iter : int, default=1000
            Maximum number of iterations.
        **kwargs
            Additional keyword arguments passed to `LogisticRegression`.
        """
        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
        else:
            self.scaler = None
        classifiers = {
            "logistic": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "MLP": MLPClassifier,
            "SVM": SVC,
            "ada_boost": AdaBoostClassifier,
            "gradient_boosting": GradientBoostingClassifier,
        }
        if classifier not in classifiers:
            raise ValueError(f"Classifier {classifier} not supported. Choose one of {list(classifiers.keys())}")
        if classifier == "logistic" and "max_iter" not in kwargs:
            kwargs["max_iter"] = 1000
        self.classifier_name = classifier
        self.classifier = classifiers[classifier](**kwargs)
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict the labels of the test data.

        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            Predicted labels.
        """

        if self.scaler is not None:
            # if all mean parameters are between 0 and 1, then probably this is a binary classifier, and we should assert that the input features are binary
            if np.logical_and(0 <= self.scaler.mean_, self.scaler.mean_ <= 1).all():
                assert (np.logical_or(X_test == 0,
                                      X_test == 1)).all(), "Classifier is probably a binary classifier, but the input features are not binary. This assert can give false positives sometimes, feel free to comment it out."
            X_test = self.scaler.transform(X_test)
        y_pred = self.classifier.predict(X_test)
        return y_pred

    def predict_proba(self, X_test):
        """Predict the labels of the test data.

        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred_proba : array-like, shape (n_samples,)
            Predicted probabilities.
        """

        if self.scaler is not None:
            # if all mean parameters are between 0 and 1, then probably this is a binary classifier, and we should assert that the input features are binary
            if np.logical_and(0 <= self.scaler.mean_, self.scaler.mean_ <= 1).all():
                assert (np.logical_or(X_test == 0,
                                      X_test == 1)).all(), "Classifier is probably a binary classifier, but the input features are not binary. This assert can can give false positives sometimes, feel free to comment it out."
            X_test = self.scaler.transform(X_test)
        if self.classifier_name in ["logistic", "random_forest", "MLP", "ada_boost", "gradient_boosting"]:
            y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        else:
            raise ValueError(f"Classifier {self.classifier_name} does not support predict_proba.")
        return y_pred_proba

    def evaluate(self, X_test, y_test, return_ys=False):
        """Evaluate the model on test data.

        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test data.
        y_test : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        accuracy : float
            Accuracy score.
        auc : float
            AUC score.
        conf_matrix : array-like, shape (2, 2)
            Confusion matrix.
        """
        if self.scaler is not None:
            # if all mean parameters are between 0 and 1, then probably this is a binary classifier, and we should assert that the input features are binary
            if np.logical_and(0 <= self.scaler.mean_, self.scaler.mean_ <= 1).all():
                assert (np.logical_or(X_test == 0,
                                      X_test == 1)).all(), "Classifier is probably a binary classifier, but the input features are not binary. This assert can can give false positives sometimes, feel free to comment it out."
            X_test = self.scaler.transform(X_test)

        y_pred = self.classifier.predict(X_test)
        # compute probability
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        if self.classifier_name in ["logistic", "random_forest", "MLP", "ada_boost", "gradient_boosting"]:
            y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            y_pred_proba = None
            auc = None
        return_objects = [accuracy, auc, conf_matrix]
        if return_ys:
            return_objects += [y_pred, y_pred_proba]
        return return_objects


def plot_feature_importance(logreg):
    """Plot the distribution of the absolute values of the coefficients and a scatter plot showing that vs their position.

    Parameters
    ----------
    logreg : LogisticRegression
        Fitted model.

    Returns
    -------
    fig, ax : tuple of matplotlib.figure.Figure and matplotlib.axes.Axes
        Figure and axes.
    """
    abs_values = np.abs(logreg.coef_[0])
    # create two subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # create a plot showing distribution of the above values
    ax[0].hist(abs_values, bins=100)
    ax[0].set_xlabel("Absolute value of the coefficient")
    ax[0].set_ylabel("Number of features")
    # check whether the absolute value is correlated with the feature position
    ax[1].scatter(range(len(abs_values)), abs_values)
    ax[1].set_xlabel("Feature position")
    ax[1].set_ylabel("Absolute value of the coefficient")
    return fig, ax


def obtain_logreg_matrix(X_train_list, y_train_list, X_test_list, y_test_list, uniformize_n_samples=True,
                         classifier="logistic", rng=None):
    """Fit a logistic regression model for each combination of training and test sets and return the accuracy, AUC and confusion matrix.

    Parameters
    ----------
    X_train_list : list of array-like, shape (n_samples, n_features)
        Training data.
    y_train_list : list of array-like, shape (n_samples,)
        Target values.
    X_test_list : list of array-like, shape (n_samples, n_features)
        Test data.
    y_test_list : list of array-like, shape (n_samples,)
        Target values.
    uniformize_n_samples : bool, defaults to True
        If True, it will make all datasets in X_train_list (and corresponding labels in y_train_list) to be of the same
        length, discarding all samples above the length of the shortest one.
    classifier : str, default="logistic"
        The classifier to use. One of "logistic", "random_forest", "MLP", "SVM", "ada_boost", "gradient_boosting".
    rng : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    Returns
    -------
    accuracy_matrix : array-like, shape (n_train_sets, n_test_sets)
        Accuracy scores.
    auc_matrix : array-like, shape (n_train_sets, n_test_sets)
        AUC scores.
    conf_matrix_list : list of array-like, shape (n_train_sets, n_test_sets, 2, 2)
        Confusion matrices.
    logregs_list : list of LogisticRegression
        Fitted models.
    """
    assert len(X_train_list) == len(y_train_list)
    assert len(X_test_list) == len(y_test_list)
    if uniformize_n_samples:
        # find the length of the shortest dataset
        length = np.infty
        for X_train in X_train_list:
            length = min(length, len(X_train))
        # then discard all samples and labels above that
        X_train_list = [X_train[0:length] for X_train in X_train_list]
        y_train_list = [y_train[0:length] for y_train in y_train_list]

    accuracy_matrix = np.zeros((len(X_train_list), len(X_test_list)))
    auc_matrix = np.zeros((len(X_train_list), len(X_test_list)))
    conf_matrix_list = []
    logregs_list = []
    # create a double loop
    for i, (X_train, y_train) in enumerate(zip(X_train_list, y_train_list)):
        conf_matrix_list.append([])
        logregs_list.append([])
        # train the model on the training set
        logreg = Classifier(X_train, y_train, random_state=rng, classifier=classifier)
        for j, (X_test, y_test) in enumerate(zip(X_test_list, y_test_list)):
            accuracy, auc, conf_matrix = logreg.evaluate(X_test, y_test)
        conf_matrix_list[-1].append(conf_matrix)
        logregs_list[-1].append(logreg.classifier)
        accuracy_matrix[i, j] = accuracy
        auc_matrix[i, j] = auc
    return accuracy_matrix, auc_matrix, conf_matrix_list, logregs_list


def plot_matrix(matrix, xnames, ynames, size=4, cmap=None):
    xticks = [f"Test on {text}" for text in xnames]
    yticks = [f"Train on {text}" for text in ynames]
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    # add colorbar to the plot
    cbar = ax.figure.colorbar(im, ax=ax)
    # remove the grid
    ax.grid(False)
    # scale the colormap between 0 and 1
    im.set_clim(0, 1)
    # rotate x labels by 45 degrees
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # write numbers in each square
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax.text(j, i, round(matrix[i, j], 2), ha="center", va="center", color="w")
    return fig, ax


def find_best_threshold(y_pred_proba, labels, y_pred_test_proba=None, labels_test=None):
    # find the best threshold
    best_threshold = 0
    best_accuracy = 0
    for threshold in np.linspace(0, 1, 100):
        y_pred = (y_pred_proba > threshold).astype(int)
        accuracy = (y_pred == labels).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return_args = (best_threshold, best_accuracy)
    if y_pred_test_proba is not None and labels_test is not None:
        y_pred_test = (y_pred_test_proba > best_threshold).astype(int)
        test_accuracy = (y_pred_test == labels_test).mean()
        return_args += (test_accuracy,)
    return return_args


class SequentialFeatureSelectorMine(SequentialFeatureSelector):
    """Updated version of the SequentialFeatureSelector class, storing the score and the order in which new features
    are added."""

    def __init__(
            self,
            estimator,
            *,
            n_features_to_select="warn",
            tol=None,
            direction="forward",
            scoring=None,
            cv=5,
            n_jobs=None,
    ):
        super().__init__(
            estimator,
            n_features_to_select=n_features_to_select,
            tol=tol,
            direction=direction,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
        )
        self.scores_ = []
        self.order_ = []

    def fit(self, X, y=None):
        """Learn the features to select from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,), default=None
            Target values. This parameter may be ignored for
            unsupervised learning.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # self._validate_params()

        # FIXME: to be removed in 1.3
        if self.n_features_to_select in ("warn", None):
            # for backwards compatibility
            warnings.warn(
                "Leaving `n_features_to_select` to "
                "None is deprecated in 1.0 and will become 'auto' "
                "in 1.3. To keep the same behaviour as with None "
                "(i.e. select half of the features) and avoid "
                "this warning, you should manually set "
                "`n_features_to_select='auto'` and set tol=None "
                "when creating an instance.",
                FutureWarning,
            )

        tags = self._get_tags()
        X = self._validate_data(
            X,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )
        n_features = X.shape[1]

        # FIXME: to be fixed in 1.3
        error_msg = (
            "n_features_to_select must be either 'auto', 'warn', "
            "None, an integer in [1, n_features - 1] "
            "representing the absolute "
            "number of features, or a float in (0, 1] "
            "representing a percentage of features to "
            f"select. Got {self.n_features_to_select}"
        )
        if self.n_features_to_select in ("warn", None):
            if self.tol is not None:
                raise ValueError("tol is only enabled if `n_features_to_select='auto'`")
            self.n_features_to_select_ = n_features // 2
        elif self.n_features_to_select == "auto":
            if self.tol is not None:
                # With auto feature selection, `n_features_to_select_` will be updated
                # to `support_.sum()` after features are selected.
                self.n_features_to_select_ = n_features - 1
            else:
                self.n_features_to_select_ = n_features // 2
        elif isinstance(self.n_features_to_select, Integral):
            if not 0 < self.n_features_to_select < n_features:
                raise ValueError(error_msg)
            self.n_features_to_select_ = self.n_features_to_select
        elif isinstance(self.n_features_to_select, Real):
            self.n_features_to_select_ = int(n_features * self.n_features_to_select)

        if self.tol is not None and self.tol < 0 and self.direction == "forward":
            raise ValueError("tol must be positive when doing forward selection")

        cloned_estimator = clone(self.estimator)

        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection
        current_mask = np.zeros(shape=n_features, dtype=bool)
        n_iterations = (
            self.n_features_to_select_
            if self.n_features_to_select == "auto" or self.direction == "forward"
            else n_features - self.n_features_to_select_
        )

        old_score = -np.inf
        is_auto_select = self.tol is not None and self.n_features_to_select == "auto"
        for _ in range(n_iterations):
            new_feature_idx, new_score = self._get_best_new_feature_score(
                cloned_estimator, X, y, current_mask
            )
            if is_auto_select and ((new_score - old_score) < self.tol):
                break

            old_score = new_score
            current_mask[new_feature_idx] = True

            self.scores_.append(new_score)
            self.order_.append(new_feature_idx)

        if self.direction == "backward":
            current_mask = ~current_mask

        self.support_ = current_mask
        self.n_features_to_select_ = self.support_.sum()

        return self
