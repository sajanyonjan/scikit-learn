#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
#         Lars Buitinck
#
# License: BSD 3 clause

from libc.string cimport memset
import numpy as np
cimport numpy as np

cdef extern from "cblas.h":
    double cblas_dasum(int, const double *, int) nogil

ctypedef float [:, :] float_array_2d_t
ctypedef double [:, :] double_array_2d_t

ctypedef (const float) [:, :] const_float_array_2d_t
ctypedef (const double) [:, :] const_double_array_2d_t

cdef fused const_floating1d:
    (const float)[::1]
    (const double)[::1]

cdef fused floating_array_2d_t:
    float_array_2d_t
    double_array_2d_t

cdef fused const_floating_array_2d_t:
    const_float_array_2d_t
    const_double_array_2d_t


np.import_array()


def _chi2_kernel_fast(const_floating_array_2d_t X,
                      const_floating_array_2d_t Y,
                      floating_array_2d_t result):
    cdef np.npy_intp i, j, k
    cdef np.npy_intp n_samples_X = X.shape[0]
    cdef np.npy_intp n_samples_Y = Y.shape[0]
    cdef np.npy_intp n_features = X.shape[1]
    cdef double res, nom, denom

    with nogil:
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                res = 0
                for k in range(n_features):
                    denom = (X[i, k] - Y[j, k])
                    nom = (X[i, k] + Y[j, k])
                    if nom != 0:
                        res  += denom * denom / nom
                result[i, j] = -res


def _sparse_manhattan(const_floating1d X_data, const int[:] X_indices,
                      const int[:] X_indptr,
                      const_floating1d Y_data, const int[:] Y_indices,
                      const int[:] Y_indptr,
                      np.npy_intp n_features, double[:, ::1] D):
    """Pairwise L1 distances for CSR matrices.

    Usage:

    >>> D = np.zeros(X.shape[0], Y.shape[0])
    >>> sparse_manhattan(X.data, X.indices, X.indptr,
    ...                  Y.data, Y.indices, Y.indptr,
    ...                  X.shape[1], D)
    """
    cdef double[::1] row = np.empty(n_features)
    cdef np.npy_intp ix, iy, j

    with nogil:
        for ix in range(D.shape[0]):
            for iy in range(D.shape[1]):
                # Simple strategy: densify current row of X, then subtract the
                # corresponding row of Y.
                memset(&row[0], 0, n_features * sizeof(double))
                for j in range(X_indptr[ix], X_indptr[ix + 1]):
                    row[X_indices[j]] = X_data[j]
                for j in range(Y_indptr[iy], Y_indptr[iy + 1]):
                    row[Y_indices[j]] -= Y_data[j]

                D[ix, iy] = cblas_dasum(n_features, &row[0], 1)
