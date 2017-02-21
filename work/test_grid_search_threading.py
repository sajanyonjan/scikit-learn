from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs

from sklearn.externals.joblib import parallel_backend
from sklearn.base import clone
# Test grid-search on classifier that has no score function.

from sklearn.externals.joblib import Parallel, delayed

def func():
    clf = LinearSVC(random_state=0)
    X, y = make_blobs(random_state=0, centers=2)
    Cs = [.1, 1, 10]
    clf_2 = LinearSVC(random_state=0)
    # for C in Cs:
    #     cloned_clf = clone(clf)
    #     cloned_clf.set_params(C)
    #     cloned_clf.fit(X, y)
    #     print(

    with parallel_backend('threading'):
        grid_search = GridSearchCV(clf, {'C': Cs}, scoring='accuracy')
        grid_search.fit(X, y)

        grid_search_2 = GridSearchCV(clf_2, {'C': Cs},
                                     scoring='accuracy')
        # smoketest grid search
        grid_search_2.fit(X, y)

    # check that best params are equal
    print(grid_search.best_params_, grid_search_2.best_params_)
    # assert grid_search_2.best_params_ ==  grid_search.best_params_
    # check that we can call score and that it gives the correct result
    # assert_equal(grid_search.score(X, y), grid_search_no_score.score(X, y))

    # # giving no scoring function raises an error
    # grid_search_no_score = GridSearchCV(clf_no_score, {'C': Cs})
    # assert_raise_message(TypeError, "no scoring", grid_search_no_score.fit,
    #                      [[1]])


from multiprocessing.pool import ThreadPool
pool = ThreadPool(2)

clf = LinearSVC(random_state=0)
X, y = make_blobs(random_state=0, centers=2)
Cs = [.1, 1, 10]



def func(clf, X, y):
    def inner(C):
        cloned_clf = clone(clf)
        cloned_clf.set_params(C=C)
        cloned_clf.fit(X, y)
        return cloned_clf.score(X, y)
    return inner

print(pool.map(func(clf, X, y), Cs))

print(Parallel(n_jobs=2, backend='threading')(delayed(func(clf, X, y), check_pickle=False)(C) for C in Cs))
