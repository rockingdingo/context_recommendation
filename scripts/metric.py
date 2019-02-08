import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer

def dcg_score(y_true, y_score, k=5):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def ndcg_score_recommend_batch(ground_truth, predictions, k=5):
    """  
        ground_truth  [n_sample, n_list_size]  relation [0,1]
        predictions   [n_sample, n_list_size]  return score
        ground_truth = [[1, 0, 1, 0]]
        predictions = [[10.0, 8.0, 7.0, 2.0]]
        ndcg_score_recommend(ground_truth, predictions)
    """
    scores = []
    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(ground_truth, predictions):
        # check if there is positive_label in y_true
        if sum(y_true) > 0.0:
            actual = dcg_score(y_true, y_score, k)
            best = dcg_score(y_true, y_true, k)
            score = float(actual) / float(best)
            scores.append(score)
    return np.mean(scores)

def ndcg_score_recommend_single(ground_truth, predictions, k=5):
    """  
        ground_truth  [n_sample, n_list_size]  relation [0,1]
        predictions   [n_sample, n_list_size]  return score
        ground_truth = [[1, 0, 1, 0]]
        predictions = [[10.0, 8.0, 7.0, 2.0]]
        ndcg_score_recommend(ground_truth, predictions)
    """
    if sum(ground_truth) > 0.0:
        actual = dcg_score(ground_truth, predictions, k)
        best = dcg_score(ground_truth, ground_truth, k)
        score = float(actual) / float(best)
    return score

def ndcg_score(ground_truth, predictions, k=5):
    """Normalized discounted cumulative gain (NDCG) at rank K.

    Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
    recommendation system based on the graded relevance of the recommended
    entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
    ranking of the entities.

    ground_truth 
    ground_truth = [0]
    predictions = [[9.0, 0.0,2.0, 3.0]]
    
    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes]
        Predicted probabilities.
    k : int
        Rank.

    Returns
    -------
    score : float

    Example
    -------
    >>> ground_truth = [1, 0, 2]
    >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    1.0
    >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    0.6666666666
    """
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)

def test():
    ground_truth = [1.0, 0.0, 0.0, 0.0, 0.0]
    predictions = [0.3, 0.4, 0.6, 0.4, 0.0]
    ndcg = ndcg_score_recommend_single(ground_truth, predictions, k=5)
    print (ndcg)
