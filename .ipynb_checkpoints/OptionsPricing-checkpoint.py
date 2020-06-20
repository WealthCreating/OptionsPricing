#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Class implementing different options pricing techniques.
Products priced: call, put, asain, barrier and knockout options."""

class OptionsPricer:
    
    def __init__(self,S0, vol, rf, T, N, steps):
        self.S0 = S0
        self.vol = vol
        self.rf = rf
        self.T = T
        self.N = N
        self.M = steps
        
    @property 
    
    def S0(self):
        return self.__S0
    
    @property
    
    def vol(self):
        return self.__vol
    
    @property 
    
    def rf(self):
        return self.__rf
    
    @property
    
    def T(self):
        return self.__T
    
    @property
    
    def N(self):
        return self.__N
    
    @property
    
    def M(self):
        return self.__M
    
    @S0.setter 
    
    def S0(self, S0):
        self.__S0 = S0
    
    @vol.setter
    
    def vol(self, vol):
        self.__vol = vol
    
    @rf.setter 
    
    def rf(self, rf):
        self.__rf = rf 
    
    @T.setter
    
    def T(self, T):
        self.__T = T
    
    @N.setter
    
    def N(self, N):
        self.__N = N
        
    @M.setter
    
    def M(self, steps):
        self.__M = steps
        
    """Matrix of storing paths"""
    
    def paths(self):
        matrixPaths = np.zeros((self.N, self.M))
        matrixPaths[0,:] = self.S0
        return matrixPaths

