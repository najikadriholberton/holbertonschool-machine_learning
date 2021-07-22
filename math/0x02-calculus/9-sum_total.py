#!/usr/bin/env python3

"""
i2 summation module
"""


def summation_i_squared(n):
    """summation of i2 using a summation formula"""
    if n < 1 or type(n) != int:
        return None
    return int((n*(n+1)*(2*n+1))/6)
