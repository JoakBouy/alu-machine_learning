#!/usr/bin/env python3
"""
Module that concatenates two matrices
"""


def matrix_shape(matrix):
    """
        Get the matrix shape
    """
    matrix_shape = []
    while (type(matrix) is list):
        matrix_shape.append(len(matrix))
        matrix = matrix[0]
    return matrix_shape


def cat_matrices(mat1, mat2, axis=0):
    """
        concatenate a matrix
    """
    from copy import deepcopy
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if len(shape1) != len(shape2):
        return None
    for i in range(len(shape1)):
        if i != axis:
            if shape1[i] != shape2[i]:
                return None
    return rec(deepcopy(mat1), deepcopy(mat2), axis, 0)


def rec(m1, m2, axis=0, current=0):
    """
        Do some reclusive calling
    """
    if axis != current:
        return [rec(m1[i], m2[i], axis, current + 1) for i in range(len(m1))]
    m1.extend(m2)
    return m1
