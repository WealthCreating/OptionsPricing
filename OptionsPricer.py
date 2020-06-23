#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


"""Class implementing options pricing based on Monte-Carlo simulations.
Products priced: call, put, asain, barrier and knockout options."""

class OptionsPricer:
    
    """Class to be used with GenerateStockPricePaths"""
    """stockPaths: DataFrame of stock path scenarios
       rf: risk free rate
       T: maturity
       K: strike"""
    
    def __init__(self, stockPaths, rf, K, T):
        self.stockPaths = stockPaths
        self.rf = rf
        self.T = T
        self.K = K
        
    @property 
    
    def stockPaths(self):
        return self.__stockPaths
    
    @property
    
    def rf(self):
        return self.__rf
    
    
    @property
    
    def K(self):
        return self.__K
    
    
    @property
    
    def T(self):
        return self.__T
        
        
    @stockPaths.setter 
    
    def stockPaths(self, stockPaths):
        self.__stockPaths = stockPaths
    
    @rf.setter
    
    def rf(self,rf):
        self.__rf = rf
        
    @K.setter
    
    def K(self,K):
        self.__K = K
        
        
    @T.setter
    
    def T(self,T):
        self.__T = T    
        
    """Price of European call option"""

    def callEU(self):
        finalPrice = self.stockPaths.iloc[:,-1]
        payoff = [max((i-self.K),0) for i in finalPrice]
        price = sum(payoff)/self.stockPaths.shape[0]
        priceDiscounted = price*np.exp(-self.rf*self.T)
        return priceDiscounted
    
    
    """Price of European put option"""

    def putEU(self):
        finalPrice = self.stockPaths.iloc[:,-1]
        payoff = [max((self.K-i),0) for i in finalPrice]
        price = sum(payoff)/self.stockPaths.shape[0]
        priceDiscounted = price*np.exp(-self.rf*self.T)
        return priceDiscounted
    
    
    """Price of Asian call option"""

    def callAsian(self):
        payoff = [max((np.mean(self.stockPaths.iloc[i,:])-self.K),0) for i in range(self.stockPaths.shape[0])]
        price = sum(payoff)/self.stockPaths.shape[0]
        priceDiscounted = price*np.exp(-self.rf*self.T)
        return priceDiscounted
    
    
    """Price of Asian put option"""

    def putAsian(self):
        payoff = [max((self.K-np.mean(self.stockPaths.iloc[i,:])),0) for i in range(self.stockPaths.shape[0])]
        price = sum(payoff)/self.stockPaths.shape[0]
        priceDiscounted = price*np.exp(-self.rf*self.T)
        return priceDiscounted
    
    
    """Price of Up-and-out Knockout call option"""
    
    def callUpAndOutKnock(self, barrier):
        stockPathsReachedBarrier = self.stockPaths >= barrier
        payoff = []
        for i in range(self.stockPaths.shape[0]):
            if stockPathsReachedBarrier.iloc[i,:].isin([True]).sum() == 0:
                payoff.append(max((self.stockPaths.iloc[i,-1]-self.K),0))
        price = sum(payoff)/self.stockPaths.shape[0]
        priceDiscounted = price*np.exp(-self.rf*self.T)
        return priceDiscounted
    
    
    """Price of Up-and-in Knockout call option"""

    def callUpAndInKnock(self, barrier):
        stockPathsReachedBarrier = self.stockPaths >= barrier
        payoff = []
        for i in range(self.stockPaths.shape[0]):
            if stockPathsReachedBarrier.iloc[i,:].isin([True]).sum() != 0:
                payoff.append(max((self.stockPaths.iloc[i,-1]-self.K),0))
        price = sum(payoff)/self.stockPaths.shape[0]
        priceDiscounted = price*np.exp(-self.rf*self.T)
        return priceDiscounted
    
    
    """Price of Down-and-out Knockout call option"""

    def callDownAndOutKnock(self, barrier):
        stockPathsReachedBarrier = self.stockPaths <= barrier
        payoff = []
        for i in range(self.stockPaths.shape[0]):
            if stockPathsReachedBarrier.iloc[i,:].isin([True]).sum() == 0:
                payoff.append(max((self.stockPaths.iloc[i,-1]-self.K),0))
        price = sum(payoff)/self.stockPaths.shape[0]
        priceDiscounted = price*np.exp(-self.rf*self.T)
        return priceDiscounted
    
    
    
    """Price of Down-and-in Knockout call option"""

    def callDownAndInKnock(self, barrier):
        stockPathsReachedBarrier = self.stockPaths <= barrier
        payoff = []
        for i in range(self.stockPaths.shape[0]):
            if stockPathsReachedBarrier.iloc[i,:].isin([True]).sum() != 0:
                payoff.append(max((self.stockPaths.iloc[i,-1]-self.K),0))
        price = sum(payoff)/self.stockPaths.shape[0]
        priceDiscounted = price*np.exp(-self.rf*self.T)
        return priceDiscounted
    
    
    
    """Price of Up-and-out Knockout put option"""

    def putUpAndOutKnock(self, barrier):
        stockPathsReachedBarrier = self.stockPaths >= barrier
        payoff = []
        for i in range(self.stockPaths.shape[0]):
            if stockPathsReachedBarrier.iloc[i,:].isin([True]).sum() == 0:
                payoff.append(max((self.K-self.stockPaths.iloc[i,-1]),0))
        price = sum(payoff)/self.stockPaths.shape[0]
        priceDiscounted = price*np.exp(-self.rf*self.T)
        return priceDiscounted
       
    
    
    """Price of Up-and-in Knockout put option"""

    def putUpAndInKnock(self, barrier):
        stockPathsReachedBarrier = self.stockPaths >= barrier
        payoff = []
        for i in range(self.stockPaths.shape[0]):
            if stockPathsReachedBarrier.iloc[i,:].isin([True]).sum() != 0:
                payoff.append(max((self.K-self.stockPaths.iloc[i,-1]),0))
        price = sum(payoff)/self.stockPaths.shape[0]
        priceDiscounted = price*np.exp(-self.rf*self.T)
        return priceDiscounted
    
    
    """Price of Down-and-out Knockout put option"""

    def putDownAndOutKnock(self, barrier):
        stockPathsReachedBarrier = self.stockPaths <= barrier
        payoff = []
        for i in range(self.stockPaths.shape[0]):
            if stockPathsReachedBarrier.iloc[i,:].isin([True]).sum() == 0:
                payoff.append(max((self.K-self.stockPaths.iloc[i,-1]),0))
        price = sum(payoff)/self.stockPaths.shape[0]
        priceDiscounted = price*np.exp(-self.rf*self.T)
        return priceDiscounted   
    
    
    """Price of Down-and-in Knockout put option"""

    def putDownAndInKnock(self, barrier):
        stockPathsReachedBarrier = self.stockPaths >= barrier
        payoff = []
        for i in range(self.stockPaths.shape[0]):
            if stockPathsReachedBarrier.iloc[i,:].isin([True]).sum() != 0:
                payoff.append(max((self.K-self.stockPaths.iloc[i,-1]),0))
        price = sum(payoff)/self.stockPaths.shape[0]
        priceDiscounted = price*np.exp(-self.rf*self.T)
        return priceDiscounted   

