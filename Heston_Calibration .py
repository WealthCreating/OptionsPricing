#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import zipfile
import os
from scipy import stats
from scipy import interpolate
from scipy.optimize import minimize
import datetime as dt
import matplotlib.pyplot as plt
import re


# # Importation of dataset (8 month worth of options prices)

# In[2]:


#path = "Insert the location of cboe-data.zip here"
def read_cboe_data(zip_file_name):
   zip_ref = zipfile.ZipFile(zip_file_name)
   zip_ref.extractall('.')
   zip_ref.close()
entire_dataset = []
#for fname in glob.glob(path):
    #read_cboe_data(fname)
    #entire_dataset.append(pd.read_csv(fname))


# In[3]:


entire_dataset[0]


# # Definition of functions needed for the calibration of the model

# In[4]:


def chct_fctH(sig,eta,rho,kappa,theta,u,S,r,q,T):
    #Definition of d and g used within the characteristic function of the Heston model
    d = np.sqrt((rho*theta*u*(1j)-kappa)**2-(theta**2)*(-(1j*u)-u**2))
    g = (kappa-rho*theta*u*(1j)-d)/(kappa-rho*theta*u*(1j)+d)
    bit_1 = np.exp((1j)*u*(np.log(S)+(r-q)*T))
    helper = (kappa-rho*theta*u*(1j)-d)
    helper2 = (1-g*np.exp(-d*T))/(1-g)
    helper3 = (1-np.exp(-d*T))/(1-g*np.exp(-d*T))
    bit_2 = np.exp(eta*kappa/theta**2*(helper*T-2*np.log(helper2)))
    bit_3 = np.exp(sig**2/theta**2*helper*helper3)
    return bit_1*bit_2*bit_3

def fast_pricing_vanilla_Heston(sig_,level_mean,corr_vs,spread_mean,theta_,p,strike,r_,q_,t,call_put):
    eta = 0.25
    N = 4096.0
    alpha = 1.5
    lam_da = 2*np.pi/(N*eta)
    b = lam_da*N/2
    strikes = np.arange(-b,b,lam_da)
    KK = np.exp(strikes) #Factor present in Carr-Madan formula
    v = np.arange(0, N*eta, eta)
    u_ = v - (alpha+1)*1j
    #################################
    ## Carr-Madan extension for accuracy gain
    #################################
    sw = (3+(-1)**np.arange(1,N+1,1))
    sw[0] = 1
    sw = sw/3
    #################################
    rho = np.exp(-r_*t)*chct_fctH(sig_,level_mean,corr_vs,spread_mean,theta_,u_,p,r_,q_,t)/(alpha**2+alpha-v**2+1j*(2*alpha+1)*v)
    A = rho*np.exp(1j*v*b)*eta*sw
    Z = np.real(np.fft.fft(A))
    CallPricesHS = (np.exp(-alpha*strikes)/np.pi)*Z
    spline_rep = interpolate.splrep(KK, CallPricesHS, s=0) #Parametrisation of the curve used to interpolate the data
    CallPriceHSFFT = interpolate.splev(strike, spline_rep, der=0) #Proper cubic interpolation
    PutPriceHSFFT = CallPriceHSFFT-p*np.exp(-q_*t)+strike*np.exp(-r_*t)
    return np.where(call_put, CallPriceHSFFT, PutPriceHSFFT)

def RMSE(vector_initial,*args):
    error = 0
    for i in range(0,args[3].size):
        rates = interpolate.splev(args[3][i], args[4], der=0)
        args2 = (args[0][i].mkt_price.values,args[1][0],
                args[0][i].strike.values,rates,args[2][i],args[3][i],args[0][i].cop)
        theo_price = fast_pricing_vanilla_Heston(vector_initial[0],
                                                     vector_initial[1],
                                                     vector_initial[2],
                                                     vector_initial[3],
                                                     vector_initial[4], 
                                                     *args2[1:])
        error += np.sum((theo_price - args2[0])**2)
    return np.sqrt(error/args[5])


# # Model Calibration

# In[5]:


def Calibration(FrameData):    
    FrameData = FrameData[(FrameData.underlying_symbol == "^SPX")]
    #Date format
    date_format = "%Y-%m-%d"
    #Data cleaning
    spot_ = np.array(FrameData["underlying_bid_eod"]) #S0 - Same across the option chain dataset
    FrameData["cop"] = np.logical_not(FrameData["option_type"] == "p")
    FrameData = FrameData[["quote_date","expiration","strike","bid_1545","ask_1545","underlying_bid_eod","cop"]]
    FrameData["mkt_price"] = FrameData.apply(lambda helper: (helper.bid_1545+helper.ask_1545)*0.5, axis = 1)
    helper3 = pd.to_datetime(FrameData["expiration"])
    helper4 = pd.to_datetime(FrameData["quote_date"])
    FrameData = FrameData[helper3-helper4 >= "7 days"]
    helper3 = pd.to_datetime(FrameData["expiration"])
    helper4 = pd.to_datetime(FrameData["quote_date"])
    mat = (helper3-helper4)/np.timedelta64(1, 'D')
    mat = np.array(mat/365)
    FrameData["maturity"] = mat
    columns = ["quote_date","expiration","bid_1545","ask_1545"]
    FrameData = FrameData.drop(columns, axis = 1)
    #Group of different strikes with the same maturity - identical call/put
    mat_unique_ = np.unique(mat)
    mat_unique_strike_ = []
    for i in range(mat_unique_.size):
        mat_unique_strike_.append((FrameData.loc[FrameData["maturity"] == mat_unique_[i]]))
    #Interpolation to derive interest rates from term structure
    rates = np.array([0.0212,0.0296,0.0308,0.0322])
    time = np.array([3/12,5,10,30])
    fct_rates = interpolate.splrep(time,rates, s=0)

    #Piece of code selecting OTM options
    otm_options = []
    for i in range(mat_unique_.size):
        cond1 = mat_unique_strike_[i].strike<mat_unique_strike_[i].underlying_bid_eod
        cond2 = mat_unique_strike_[i].cop == True
        cond3 = mat_unique_strike_[i].strike>mat_unique_strike_[i].underlying_bid_eod
        cond4 = mat_unique_strike_[i].cop == False
        #otm_options = list of data frames with OTM options grouped by maturity
        otm_options.append(mat_unique_strike_[i].drop(mat_unique_strike_[i][(cond1&cond2)|(cond3&cond4)].index))
    
    #Estimation the dividend yield solving the Put-Call parity for q
    forward_contract = []
    interest_r = []
    mat_unique_strike_og_c = []
    mat_unique_strike_og_p = []
    div_yield = []
    yields = 0.02*np.ones(mat_unique_.size)
    for i in range(mat_unique_.size):
        #mat_unique_strike_og_c/p = list of OTM, ATM and ITM options grouped by maturity 
        mat_unique_strike_og_c.append(mat_unique_strike_[i][mat_unique_strike_[i]["cop"]==True]) 
        mat_unique_strike_og_p.append(mat_unique_strike_[i][mat_unique_strike_[i]["cop"]==False])
        forward_contract.append(mat_unique_strike_og_c[i].mkt_price.values - mat_unique_strike_og_p[i].mkt_price.values)
        interest_r.append((forward_contract[i]+mat_unique_strike_og_c[i].strike*np.exp(-interpolate.splev(mat_unique_[i], fct_rates, der=0)*mat_unique_[i]))/spot_[0] + interpolate.splev(mat_unique_[i], fct_rates, der=0))
        div_yield.append(sum(interest_r[i])/(100*len(interest_r[i])))
    
    
    #Initial values chosen for the Heston calibration
    sig_ = 0.2
    eta_ = 0.05
    rho_ = -0.7
    kappa_ = 0.05
    theta_ = 0.1
    initial_values = [sig_,eta_,rho_,kappa_,theta_]
    #ITM Options split between Calls and Puts
    for i in range(mat_unique_.size):
        mat_unique_strike_og_c[i] = otm_options[i][otm_options[i]["cop"]==True]
        mat_unique_strike_og_p[i] = otm_options[i][otm_options[i]["cop"]==False]
    
    #Number of options in different subsets
    N = 0;
    for i in range(mat_unique_.size):
        N += mat_unique_strike_og_c[i].strike.size
    
    #Optimisation problem for calls only   
    bound = [(0.0001, 1), (0.0001,5),(-1,1),(0.0001,None),(0.0001,1)]
    result = minimize(RMSE,initial_values,args=(mat_unique_strike_og_c,spot_,yields,mat_unique_,fct_rates,N),
                  bounds = bound,tol = 0.001)
    helper = [result.x[0],result.x[1],result.x[2],result.x[3],result.x[4]];
    heston_price_pc1_c = []
    heston_price_pc1_p = []
    heston_price = []
    for i in range(mat_unique_.size):
        rate = interpolate.splev(mat_unique_[i], fct_rates, der=0)
        y_c = fast_pricing_vanilla_Heston(result.x[0],result.x[1],result.x[2],result.x[3],result.x[4],spot_[0],
                                    mat_unique_strike_og_c[i].strike.values,rate,yields[i],mat_unique_[i],
                                    mat_unique_strike_og_c[i].cop.values)
        y_p = fast_pricing_vanilla_Heston(result.x[0],result.x[1],result.x[2],result.x[3],result.x[4],spot_[0],
                                    mat_unique_strike_og_p[i].strike.values,rate,yields[i],mat_unique_[i],
                                    mat_unique_strike_og_p[i].cop.values)
        y = fast_pricing_vanilla_Heston(result.x[0],result.x[1],result.x[2],result.x[3],result.x[4],spot_[0],
                                    otm_options[i].strike.values,rate,yields[i],mat_unique_[i],
                                    otm_options[i].cop.values)
        heston_price_pc1_c.append(y_c)
        heston_price_pc1_p.append(y_p)
        heston_price.append(y)
        mat_unique_strike_og_c[i]["Heston_price_pc1"] = heston_price_pc1_c[i]
        mat_unique_strike_og_p[i]["Heston_price_pc1"] = heston_price_pc1_p[i]
        otm_options[i]["Heston_price"] = heston_price[i]
    
    for i in range(mat_unique_.size):
        #Subplot for calls
        plt.subplot(1, 3, 1)
        x_heston_pc1_c = mat_unique_strike_og_c[i].mkt_price
        y_heston_pc1_c = mat_unique_strike_og_c[i].Heston_price_pc1
        plt.scatter(x_heston_pc1_c,y_heston_pc1_c, c = "b")
        x_c = mat_unique_strike_og_c[i].mkt_price
        y_c = mat_unique_strike_og_c[i].mkt_price
        plt.scatter(x_c,y_c, c = "g")
        plt.xlabel("Prices")
        plt.ylabel("Prices")
        plt.title("Calls")
        #Subplot for puts
        plt.subplot(1, 3, 2)
        x_heston_pc1_p = mat_unique_strike_og_p[i].mkt_price
        y_heston_pc1_p = mat_unique_strike_og_p[i].Heston_price_pc1
        plt.scatter(x_heston_pc1_p,y_heston_pc1_p, c = "b")
        x_p = mat_unique_strike_og_p[i].mkt_price
        y_p = mat_unique_strike_og_p[i].mkt_price
        plt.scatter(x_p,y_p, c = "g")
        plt.xlabel("Prices")
        plt.ylabel("Prices")
        plt.title("Puts")
        #Subplot for both calls and puts
        plt.subplot(1, 3, 3)
        mate2 = otm_options[i].strike
        chap2 = otm_options[i].mkt_price
        plt.scatter(mate2,chap2, c ="g")
        x_strike = otm_options[i].strike
        y_check = otm_options[i].Heston_price
        plt.scatter(x_strike,y_check, c = "b")
        plt.title("Calls & Puts")
        plt.xlabel("Strikes")
        plt.ylabel("Prices")
        plt.show()
    for i in range(mat_unique_.size):
        check_strike = otm_options[i].strike
        check_price = otm_options[i].Heston_price
        plt.scatter(check_strike,check_price,c="b")
        check_mark = otm_options[i].mkt_price
        plt.scatter(check_strike,check_mark, c="g")
        plt.title("Option Prices vs. Strikes")
        plt.xlabel("Strikes")
        plt.ylabel("Prices")
    plt.show()
    return helper


# # Output

# In[ ]:


#Actual Calibration
if __name__ == "__main__":
    list_parameters = []
    for i in range(len(entire_dataset)):
        list_parameters.append(Calibration(entire_dataset[i]));
#Printing out the data frame with the estimated parameters for each days
    summary = pd.DataFrame(list_parameters)
    columns_ = ["sigma","eta","rho","kappa","theta"]
    summary.columns = columns_
    rows_ = []
    for fname in glob.glob(path):
        match = re.search("\d{4}-\d{2}-\d{2}", fname)
        date = dt.datetime.strptime(match.group(), '%Y-%m-%d')
        date = date.strftime("%B %d, %Y")
        rows_.append(date)
    summary.index = rows_
    summary


# In[ ]:




