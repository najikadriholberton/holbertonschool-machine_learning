#!/usr/bin/python3
"""Size Me Please"""


def matrix_shape(matrix):
    """Matrix Shape function"""
    dimensions = []
    current_depth = matrix
    while type(current_depth) is list:
        dimensions.append(len(current_depth))
        current_depth = current_depth[0]
    return dimensions
