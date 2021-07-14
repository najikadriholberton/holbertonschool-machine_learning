#!/usr/bin/env python3

"""
Module for Line Up
"""


def add_arrays(arr1, arr2):
    """add two arrays element wise"""
    n = len(arr1)
    m = len(arr2)
    if n != m:
        return None
    return [arr1[i] + arr2[i] for i in range(n)]
