#!/usr/bin/python3

"""
Module for Ridin' Bareback
"""


def mat_mul(mat1, mat2):
    """Matrix Multiplication"""
    n = len(mat1)
    m1 = len(mat1[0])
    m2 = len(mat2)
    k = len(mat2[0])

    if m1 != m2:
        return None

    mat = []

    for i in range(n):
        mat.append([])
        for j in range(k):
            total_val = 0
            for l in range(m1):
                total_val += mat1[i][l] * mat2[l][j]
            mat[i].append(total_val)
    return mat
