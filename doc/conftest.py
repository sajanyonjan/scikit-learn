from os.path import exists
from os.path import join

import numpy as np

from sklearn.utils.testing import SkipTest
from sklearn.utils.testing import check_skip_network
from sklearn import datasets
from sklearn.utils.testing import install_mldata_mock
from sklearn.utils.testing import uninstall_mldata_mock


def setup_labeled_faces():
    try:
        datasets.fetch_lfw_people(download_if_missing=False)
    except OSError:
        raise SkipTest("Skipping dataset loading doctests")



def setup_mldata():
    # setup mock urllib2 module to avoid downloading from mldata.org
    install_mldata_mock({
        'mnist-original': {
            'data': np.empty((70000, 784)),
            'label': np.repeat(np.arange(10, dtype='d'), 7000),
        },
        'iris': {
            'data': np.empty((150, 4)),
        },
        'datasets-uci-iris': {
            'double0': np.empty((150, 4)),
            'class': np.empty((150,)),
        },
    })


def teardown_mldata():
    uninstall_mldata_mock()


def setup_rcv1():
    check_skip_network()
    # skip the test in rcv1.rst if the dataset has not already been downloaded
    try:
        datasets.fetch_rcv1(download_if_missing=False)
    except OSError:
        raise SkipTest("Skipping dataset loading doctests")

def setup_twenty_newsgroups():
    try:
        datasets.fetch_20newsgroups(download_if_missing=False)
    except OSError:
        raise SkipTest("Skipping dataset loading doctests")


def setup_working_with_text_data():
    check_skip_network()


def pytest_runtest_setup(item):
    fname = item.fspath.strpath
    if fname.endswith('datasets/labeled_faces.rst'):
        setup_labeled_faces()
    elif fname.endswith('datasets/mldata.rst'):
        setup_mldata()
    elif fname.endswith('datasets/rcv1.rst'):
        setup_rcv1()
    elif fname.endswith('datasets/twenty_newsgroups.rst'):
        setup_twenty_newsgroups()
    elif fname.endswith('tutorial/text_analytics/working_with_text_data.rst'):
        setup_working_with_text_data()


def pytest_runtest_teardown(item):
    fname = item.fspath.strpath
    if fname.endswith('datasets/mldata.rst'):
        teardown_mldata()
