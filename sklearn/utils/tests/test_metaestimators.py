from sklearn.utils.testing import assert_true, assert_false
from sklearn.utils.metaestimators import if_delegate_has_method


class Prefix(object):
    def func(self):
        pass


class MockMetaEstimator(object):
    # --
    a_prefix = Prefix()

    @if_delegate_has_method(delegate="a_prefix")
    def func(self):
        "This is a mock delegated function"


def test_delegated_docstring():
    assert_true("This is a mock delegated function"
                in str(MockMetaEstimator.__dict__['func'].__doc__))
    assert_true("This is a mock delegated function"
                in str(MockMetaEstimator.func.__doc__))
    assert_true("This is a mock delegated function"
                in str(MockMetaEstimator().func.__doc__))


class MetaEst(object):
    # --
    def __init__(self, sub_est, better_sub_est=None):
        self.sub_est = sub_est
        self.better_sub_est = better_sub_est

    @if_delegate_has_method(delegate='sub_est')
    def predict(self):
        pass


class MetaEstTestTuple(MetaEst):
    # --

    @if_delegate_has_method(delegate=('sub_est', 'better_sub_est'))
    def predict(self):
        pass


class MetaEstTestList(MetaEst):
    # --

    @if_delegate_has_method(delegate=['sub_est', 'better_sub_est'])
    def predict(self):
        pass


class HasPredict(object):
    # --

    def predict(self):
        pass


class HasNoPredict(object):
    # --
    pass


def test_if_delegate_has_method():
    assert_true(hasattr(MetaEst(HasPredict()), 'predict'))
    assert_false(hasattr(MetaEst(HasNoPredict()), 'predict'))
    assert_false(
        hasattr(MetaEstTestTuple(HasNoPredict(), HasNoPredict()), 'predict'))
    assert_true(
        hasattr(MetaEstTestTuple(HasPredict(), HasNoPredict()), 'predict'))
    assert_false(
        hasattr(MetaEstTestTuple(HasNoPredict(), HasPredict()), 'predict'))
    assert_false(
        hasattr(MetaEstTestList(HasNoPredict(), HasPredict()), 'predict'))
    assert_true(
        hasattr(MetaEstTestList(HasPredict(), HasPredict()), 'predict'))
