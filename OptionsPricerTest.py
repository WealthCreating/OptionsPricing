#!/usr/bin/env python
# coding: utf-8

# In[1]:


from OptionsPricer import OptionsPricer
from GenerateStockPricePaths import GenerateStockPricePaths
import unittest as ut


# In[2]:


"""Test class aiming to test OptionsPricer class."""

class OptionsPricerTest(OptionsPricer, ut.TestCase):
    def __init_(self, stockPaths, rf, K, T):
        OptionsPricer.__init__(stockPaths, rf, K, T)
        ut.TestCase.__init__(self)


# In[3]:


"""Functions used in the tests to be used in the assertEqual method."""
from scipy.stats import norm
import numpy as np
def BSClosedForm(ST,K,rf,vol,T, option = "call"):
    num_d1 = np.log(ST/K)+(rf+vol**2/2)*T
    denom_d1 = vol*np.sqrt(T)
    d1 = num_d1/denom_d1
    d2 = d1-vol*np.sqrt(T)
    discountedK = K*np.exp(-rf*T)
    callPrice = norm.cdf(d1)*ST-norm.cdf(d2)*discountedK
    if option == "call":
        return callPrice
    else:
        """Derive price of put using put-call parity"""
        putPrice = discountedK-ST+callPrice
        return putPrice


# I used Antithetic Sampling technique in order to improve my Monte-Carlo estimations.
# However, there is still some room for improvement when it comes to the convergence of error rates.

# In[4]:


if __name__ == "__main__":
    S0_ = 100
    vol_ = 0.2
    rf_ = 0.05
    T_ = 1
    K_ = 100
    stockPrices = GenerateStockPricePaths(S0_,vol_,rf_, N = 500000)
    BSStockPrices = stockPrices.BSPaths()
    test = OptionsPricerTest(BSStockPrices, rf = rf_, K = K_ ,T = T_)
    """Series of unit tests that check methods in the OptionsPricer class"""
    test.assertAlmostEqual(test.callEU(),BSClosedForm(S0_,K_,rf_,vol_,T_), places = 1)
    test.assertAlmostEqual(test.putEU(),BSClosedForm(S0_,K_,rf_,vol_,T_,option="put"), places = 1)

