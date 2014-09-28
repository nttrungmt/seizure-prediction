__author__ = 'udi'
import numpy as np
from sklearn.ensemble.gradient_boosting import BinomialDeviance


class WeightedBinomialDeviance(BinomialDeviance):
    """Binomial (two-class) deviance loss function with weights for GradientBoostingClassifier.

        sample_weight : array-like, shape = [n_samples] or None.
            If None, then samples are equally weighted.
            n_samples must match the size of X and y passed to GradientBoostingClassifier.fit

        Usage:
        >>> # create classifier object and set parameters
        >>> clf = GradientBoostingClassifier(...)
        >>> # set the loss object
        >>> clf.loss__ = WeightedBinomialDeviance(sample_weight)
    """
    def __init__(self, sample_weight=None):
        self.weights = sample_weight
        super(WeightedBinomialDeviance, self).__init__(2)

    def __call__(self, y, pred):
        """Compute the deviance (= 2 * negative log-likelihood). """
        # logaddexp(0, v) == log(1.0 + exp(v))
        pred = pred.ravel()
        return -2.0 * np.average((y * pred) - np.logaddexp(0.0, pred), weights=self.weights)

    def negative_gradient(self, y, pred, **kargs):
        """Compute the residual (= negative gradient). """
        if self.weights is None:
            return super(WeightedBinomialDeviance, self).negative_gradient(y, pred)
        else:
            return super(WeightedBinomialDeviance, self).negative_gradient(y, pred) * self.weights

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):
        """Make a single Newton-Raphson step.

        our node estimate is given by:

            sum(weight*(y - prob)) / sum(weight*prob * (1 - prob))

        we take advantage that: weight*(y - prob) = residual
        """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)

        numerator = residual.sum()
        if self.weights is None:
            denominator = np.sum((y - residual) * (1 - y + residual))
        else:
            weights = self.weights.take(terminal_region, axis=0)
            denominator = np.sum(weights * (y - residual) * (1 - y + residual))

        if denominator == 0.0:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator


class NAUC(BinomialDeviance):
    """
    ##Negative AUC loss (1-AUC Loss):
    AUC is a special cost function that can not be broken into sum of loss functions.
    However it can be approximated: http://bioinformatics.oxfordjournals.org/content/21/24/4356.full
    original code: http://homepage.stat.uiowa.edu/~jian/class/main.html
    R package: http://artax.karlin.mff.cuni.cz/r-help/library/mboost/html/Family.html
    R code: https://r-forge.r-project.org/scm/viewvc.php/pkg/R/family.R?view=markup&root=mboost&pathrev=406

    Binary classification is a special case; here, we only need to
    fit one tree instead of ``n_classes`` trees.
    """
    def __call__(self, y, pred):
        """Compute the cost/risk """
        self.negative_gradient(y, pred)
        return 1. - np.sum(self.M0 > 0)/(self.n1 * self.n0)

    @staticmethod
    def approx_grad(x):
        return np.clip(1 - np.abs(x),0.,1.)

    @staticmethod
    def approx_loss(x):
        xsign = x >= 0
        ret = xsign - (xsign - 0.5) * np.square(1-np.abs(x))
        return np.where(x < -1, 0, np.where(x > 1, 1, ret))

    def negative_gradient(self, y, pred, **kargs):
        """Compute the residual (= negative gradient). """
        self.ind1 = np.where(y==1)[0]
        self.n1 = len(self.ind1)
        self.ind0 = np.where(y==0)[0]
        self.n0 = len(self.ind0)
        # our predictions are between [0,1] but the algorithm expects [-1,1]
        pred = (pred - 0.5) / np.sd(pred)

        self.M0 = np.repeat(pred[self.ind1], self.n0) - pred[self.ind0]
        M1 = self.approx_grad(self.M0)
        ng = np.empty(self.n0 + self.n1)
        ng[self.ind1] = np.sum(M1, axis=1)
        ng[self.ind0] = np.sum(M1, axis=0)
        return ng

    def loss(self, y, pred):
        self.negative_gradient(y, pred)
        return 1. - np.sum(self.approx_loss(self.M0))/(self.n1 * self.n0)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):
        """Make a single gradient decent


        we take advantage that: residual is negative gradient
        """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)

        numerator = residual.sum()

        tree.value[leaf, 0, 0] = numerator
