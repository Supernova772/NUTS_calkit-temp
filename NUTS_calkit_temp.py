import math
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sympy import symbols, solve
import os

def import_time(mdbname):
    m=0
    with open('D:\\Temp\\'+mdbname+'Time.csv', 'r') as f:
        csvreader = csv.reader(f)  
        for row in csvreader:
            m+=1
    my_time=[0]*m
    m=0
    #
    with open('D:\\Temp\\'+mdbname+'Time.csv', 'r') as f:
        csvreader = csv.reader(f)  
        for row in csvreader:
            val=[0]*len(row)
            for i in range(len(row)):
                a=row[i]
                val[i]=float(a)
                if m==0:
                    my_time[i]=val[i]
            m=m+1
    return my_time

def import_mdbname(mdbname):
    m=0
    with open('D:\\Temp\\'+mdbname+'.csv', 'r') as f:
        csvreader = csv.reader(f)  
        for row in csvreader:
            m=m+1
    my_depth=[0]*m
    my_width=[0]*m
    my_t_f=[0]*m  
    my_t_w=[0]*m  
    my_d_p=[0]*m  
    my_lamda_p=[0]*m
    m=0
    with open('D:\\Temp\\'+mdbname+'.csv', 'r') as f:
        csvreader = csv.reader(f)  
        for row in csvreader:
            val=[0]*len(row)
            for i in range(len(row)):
                a=row[i]
                val[i]=float(a)
            my_depth[m]=val[0]
            my_width[m]=val[1]
            my_t_f[m]=val[2]
            my_t_w[m]=val[3]
            my_d_p[m]=val[4]
            my_lamda_p[m]=val[5]
            m=m+1
    return my_depth, my_width, my_t_f, my_t_w, my_d_p, my_lamda_p, m

def temp_cal(time,d,w,t_f,t_w,d_p,lamda_p):
    time=np.linspace(0,time,int(time/50))
    time=list(float(x) for x in time)
    nom_t_a_l=[0]*len(time)
    nom_t_a_u=[0]*len(time) 
    t_a_l=[0]*len(time) 
    t_a_u=[0]*len(time)
    t_a_w=[0]*len(time)
    lamda_p=10**-3*lamda_p
    t_o=[20,400.042,511.363,573.276,613.413,648.226,676.455,698.798,719.587,736.827,750.978,765.204,779.789,791.65,800.858,813.575,820.151,830.085,840.383,847.181,854.196,861.373,868.239,875.008,881.498,887.727,893.69,901.069,905.417,910.185,915.343,920.333,925.153,929.828,934.366,938.762,943.038,947.198,951.239,955.178,959.019,962.757,966.409,969.975,973.454,976.857,980.186,983.438,986.624,989.745,992.799,995.794,998.732,1001.61,1004.43,1007.21,1009.93,1012.6,1015.23,1017.81,1020.35,1022.85,1025.3,1027.72,1030.1,1032.44,1034.74,1037.01,1039.24,1041.45,1043.62,1045.76,1048.76]
    time_o=[0,97.2868,198.242,298.792,392.456,495.336,598.215,695.534,799.387,896.919,986.423,1084.65,1195.85,1294.83,1376.8,1498.92,1566.36,1673.86,1793.07,1876.45,1966.41,2062.97,2159.53,2259.53,2359.53,2459.53,2559.53,2688.63,2767.73,2857.29,2957.29,3057.29,3157.29,3257.29,3357.29,3457.29,3557.29,3657.29,3757.29,3857.29,3957.29,4057.29,4157.29,4257.29,4357.29,4457.29,4557.29,4657.29,4757.29,4857.29,4957.29,5057.29,5157.29,5257.29,5357.29,5457.29,5557.29,5657.29,5757.29,5857.29,5957.29,6057.29,6157.29,6257.29,6357.29,6457.29,6557.29,6657.29,6757.29,6857.29,6957.29,7057.29,7200]
    t_o=[20,858.7446547,937.3228412,989.2593147,1024.61115,1048.677581,1092.495753,1098.902746,1099.839562,1099.976541,1099.99657,1099.999498,1099.999927,1099.999989,1099.999998,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100]
    time_o=[0,60,120,180,240,300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600,3900,4200,4500,4800,5100,5400,5700,6000,6300,6600,6900,7200]    
    lamda_a=[0.05333,0.052,0.05067,0.04933,0.048,0.04667,0.04534,0.04401,0.04267,0.04134,0.04,0.03868,0.03734,0.03601,0.034678,0.03401,0.03252,0.03102,0.02952,0.02835,0.0273,0.0273,0.0273,0.0273,0.0273]
    t_lamda=[20,60,100,140,180,220,260,300,340,380,420,460,500,540,580,600,645,690,735,770,805,840,875,900,1200]
    c_a=[439.8,465.77,487.62,506.18,522.33,536.9,550.75,564.74,579.71,596.51,616.01,639.06,666.5,699.18,737,97,760.21,805.8,936.87,5000,1001.92,785.81,708.48,668.75,650,650]
    t_c=[20,60,100,140,180,220,260,300,340,380,420,460,5000,540,580,6000,645,690,735,770,805,840,875,900,1200]
    A_p_b = (w + t_f) * 2
    V_i_l = w * t_f
    A_p_t = w
    A_p_l = w + 2 * t_f
    A_p_u = (w + t_f) * 2
    V_i_u = w * t_f
    A_p_w = (d - 2 * t_f) * 2
    V_i_w = (d - 2 * t_f) * t_w
    d_web = (d - t_f)
    stefan=5.67e-14
    alfa_c=4e-5
    eps_m=0.8
    eps_f=1
    p_p = 0.00000024
    c_p = 1047
    p_a = 7.9e-6
    for i in range(len(time)):
        #<<<<<<<<<<<<<<< Calcualtion of first iteration >>>>>>>>>>>>>>>>>>
        if i ==0:
            #<<<<<<<<<<<<<<< Interpolation of the fire temperature >>>>>>>>>>>>>>>>>>>>>>>>>>
            for j in range(len(time_o)-1):
                if time[i] >= time_o[j] and time_o[j+1] > time[i]:
                    t_o2 = (time[i] - time_o[j]) * (t_o[j+1] - t_o[j]) / (time_o[j+1] - time_o[j]) + t_o[j]
                    break
            #Ambient temperature
            delta_t_o=t_o2
            #Simulation time
            delta_t=time[i]
            #<<<<<<<<<<<<<<< Interpolation of the upper flange nonminal specific heat  >>>>>>>>>>>>>>>>>>>>>>>>>>
            for j in range(24):
                if 20 >= t_c[j] and t_c[j+1] > 20:
                    nom_c_a_uf = (20 - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                    break
            #Interpolation for the top junction steel conductivity
            for j in range(24):
                if 20 >= t_lamda[j] and t_lamda[j+1] > 20:
                    nom_lamda_a_tj = (20- t_lamda[j]) * (lamda_a[j+1] - lamda_a[j]) / (t_lamda[j+1] - t_lamda[j]) + lamda_a[j]
                    break
            #<<<<<<<<<<<<<<<<<<<<<<<<   Interpolation of the upper flange specifica heat   >>>>>>>>>>>>>>>>>>>>>>>>>>
            for j in range(24):
                if 20 >= t_c[j] and t_c[j+1] > 20:
                    c_a_uf = (20 - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                    break
            #<<<<<<<<<<<<<<<  Interpolation of the lower flange nominal specific heat  >>>>>>>>>>>>>>>>>>>>>>>>>>
            for j in range(24):
                if 20 >= t_c[j] and t_c[j+1] > 20:
                    nom_c_a_lf = (20 - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                    break
            #Interpolation for the bottom junction steel conductivity
            for j in range(24):
                if 20 >= t_lamda[j] and t_lamda[j+1] > 20:
                    nom_lamda_a_bj = (20 - t_lamda[j]) * (lamda_a[j+1] - lamda_a[j]) / (t_lamda[j+1] - t_lamda[j]) + lamda_a[j]
                    break 
            #<<<<<<<<<<<<<<<<<<<<<<<<   Interpolation of the lower flange specifica heat   >>>>>>>>>>>>>>>>>>>>>>>>>>
            for j in range(24):
                if 20 >= t_c[j] and t_c[j+1] > 20:
                    c_a_lf = (20 - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                    break
            #<<<<<<<<<<<<<<<<<<<<<<<<   END   >>>>>>>>>>>>>>>>>>>>>>>>>>
            nom_omega_l = (c_p*p_p/nom_c_a_lf/p_a)*d_p*(A_p_b/V_i_l)
            nom_omega_u = (c_p*p_p/nom_c_a_uf/p_a)*d_p*(A_p_u/V_i_u)
            omega_l = (c_p*p_p/c_a_lf/p_a)*d_p*(A_p_b/V_i_l)
            omega_u = (c_p*p_p/c_a_uf/p_a)*d_p*(A_p_u/V_i_u)
            #<<<<<<<<<<<<<<<<<<<<<<<<  Nominal bottom flange temperature  >>>>>>>>>>>>>>>>>>>>>
            nom_delta_t_l = lamda_p/d_p/ (nom_c_a_lf* p_a*V_i_l)   * (A_p_b * 1 / (1 + nom_omega_l/ 3) * (t_o2 - 20)) * delta_t - (math.exp(nom_omega_l/ 10) - 1) * delta_t_o
            if nom_delta_t_l < 0:
                nom_delta_t_l = 0
            nom_t_a_l[i] = 20 + nom_delta_t_l
            #<<<<<<<<<<<<<<<<<<<<<<<<  Bottom flange temperature  >>>>>>>>>>>>>>>>>>>>>
            delta_t_l = lamda_p/d_p/ (c_a_lf* p_a*V_i_l)   * (A_p_b * 1 / (1 + omega_l/ 3) * (t_o2 - 20)) * delta_t - (math.exp(omega_l/ 10) - 1) * delta_t_o
            if delta_t_l < 0:
                delta_t_l = 0
            t_a_l[i] = 20 + delta_t_l
            #<<<<<<<<<<<<<<<<<<<<<<<<Nominal upper flange temperature>>>>>>>>>>>>>>>>>>>>>>>
            nom_delta_t_u = 1/ (nom_c_a_uf* p_a*V_i_u)   * (A_p_t * (alfa_c * (t_o2 - 20) + eps_m * eps_f * stefan * ((t_o2 + 273) ** 4 - (20 + 273) ** 4)) + A_p_l * 1 / (1 + nom_omega_u/3) * (lamda_p / d_p * (t_o2 - 20))) * delta_t - A_p_l / A_p_u * (math.exp(nom_omega_u/ 10) - 1) * delta_t_o
            if nom_delta_t_u < 0:
                nom_delta_t_u = 0
            nom_t_a_u[i] = 20 + nom_delta_t_u
            #<<<<<<<<<<<<<<<<<<<<<<<<Upper flange temperature>>>>>>>>>>>>>>>>>>>>>>>
            delta_t_u = 1/ (c_a_uf* p_a*V_i_u)   * (A_p_t * (alfa_c * (t_o2 - 20) + eps_m * eps_f * stefan * ((t_o2 + 273) ** 4 - (20 + 273) ** 4)) + A_p_l * 1 / (1 + omega_u/3) * (lamda_p / d_p * (t_o2 - 20))) * delta_t - A_p_l / A_p_u * (math.exp(omega_u/ 10) - 1) * delta_t_o
            if delta_t_u < 0:
                delta_t_u = 0
            t_a_u[i] = 20 + delta_t_u
    
        #<<<<<<<<<<<<<<<<<<  Calculation of the rest iterations  >>>>>>>>>>>>>>>>>>>>>>>>>        
        elif i != 0:
            #<<<<<<<<<<<<<<< Interpolation of the temperature >>>>>>>>>>>>>>>>>>>>>>>>>>
            for j in range(len(time_o)-1):
                if time[i-1] >= time_o[j] and time_o[j+1] > time[i-1]:
                    t_o1 = (time[i-1] - time_o[j]) * (t_o[j+1] - t_o[j]) / (time_o[j+1] - time_o[j]) + t_o[j]
                    break
            for j in range(len(time_o)-1):
                if time[i] >= time_o[j] and time_o[j+1] > time[i]:
                    t_o2 = (time[i] - time_o[j]) * (t_o[j+1] - t_o[j]) / (time_o[j+1] - time_o[j]) + t_o[j]
                    break
            #Temperature increment
            delta_t_o=t_o2-t_o1
            #Time increment
            delta_t=time[i]-time[i-1]
            #Interpolation for the top junction steel conductivity
            for j in range(24):
                if nom_t_a_u[i-1] >= t_lamda[j] and t_lamda[j+1] > nom_t_a_u[i-1]:
                    nom_lamda_a_tj = (nom_t_a_u[i-1]- t_lamda[j]) * (lamda_a[j+1] - lamda_a[j]) / (t_lamda[j+1] - t_lamda[j]) + lamda_a[j]
                    break
            #Interpolation for the nominal upper flange specific heat
            for j in range(24):
                if nom_t_a_u[i-1] >= t_c[j] and t_c[j+1] > nom_t_a_u[i-1]:
                    nom_c_a_uf = (nom_t_a_u[i-1] - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                    break
            #Interpolation for the upper flange specific heat    
            for j in range(24):
                if t_a_u[i-1] >= t_c[j] and t_c[j+1] > t_a_u[i-1]:
                    c_a_uf = (t_a_u[i-1] - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                    break
            #Interpolation for the bottom junction steel conductivity
            for j in range(24):
                if nom_t_a_l[i-1] >= t_lamda[j] and t_lamda[j+1] > nom_t_a_l[i-1]:
                    nom_lamda_a_bj = (nom_t_a_l[i-1] - t_lamda[j]) * (lamda_a[j+1] - lamda_a[j]) / (t_lamda[j+1] - t_lamda[j]) + lamda_a[j]
                    break          
            #Interpolation for the nominal lower flange specific heat
            for j in range(24):
                if nom_t_a_l[i-1] >= t_c[j] and t_c[j+1] > nom_t_a_l[i-1]:
                    nom_c_a_lf = (nom_t_a_l[i-1] - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                    break
            #Interpolation for the lower flange specific heat    
            for j in range(24):
                if t_a_l[i-1] >= t_c[j] and t_c[j+1] > t_a_l[i-1]:
                    c_a_lf = (t_a_l[i-1] - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                    break
            #<<<<<<<<<<<<<<<<<<<<<<<<   END   >>>>>>>>>>>>>>>>>>>>>>>>>>u
            nom_omega_l = (c_p*p_p/nom_c_a_lf/p_a)*d_p*(A_p_b/V_i_l)
            nom_omega_u = (c_p*p_p/nom_c_a_uf/p_a)*d_p*(A_p_u/V_i_u)
            omega_l = (c_p*p_p/c_a_lf/p_a)*d_p*(A_p_b/V_i_l)
            omega_u = (c_p*p_p/c_a_uf/p_a)*d_p*(A_p_u/V_i_u)            
            #<<<<<<<<<<<<<<<<<<<<<<<<  Nominal lower flange temperature  >>>>>>>>>>>>>>>>>>>>>
            nom_delta_t_l = lamda_p/d_p/ (nom_c_a_lf* p_a*V_i_l)   * (A_p_b * 1 / (1 + nom_omega_l/ 3) * (t_o2 - nom_t_a_l[i-1])) * delta_t - (math.exp(nom_omega_l/ 10) - 1) * delta_t_o
            if nom_delta_t_l < 0:
                nom_delta_t_l = 0
            nom_t_a_l[i] = nom_t_a_l[i-1] + nom_delta_t_l
            #<<<<<<<<<<<<<<<<<<<<<<<<  Bottom flange temperature  >>>>>>>>>>>>>>>>>>>>>
            delta_t_l = lamda_p/d_p/ (c_a_lf* p_a*V_i_l)   * (A_p_b * 1 / (1 + omega_l/ 3) * (t_o2 - t_a_l[i-1])+(6.9*10**-4*d-0.17)*wt*nom_lamda_a_bj/(d-t_f)*2*(nom_t_a_u[i-1]-nom_t_a_l[i-1])) * delta_t - (math.exp(omega_l/ 10) - 1) * delta_t_o
            if delta_t_l < 0:
                delta_t_l = 0
            t_a_l[i] = t_a_l[i-1] + delta_t_l
            #<<<<<<<<<<<<<<<<<<<<<<<<  Nominal upper flange temperature  >>>>>>>>>>>>>>>>>>>>>>>
            nom_delta_t_u = 1/ (nom_c_a_uf* p_a*V_i_u) * (A_p_t * (alfa_c * (t_o2 - nom_t_a_u[i-1]) + eps_m * eps_f * stefan * ((t_o2 + 273) ** 4 - (nom_t_a_u[i-1] + 273) ** 4)) + A_p_l * 1 / (1 + nom_omega_u/3) * (lamda_p / d_p * (t_o2 - nom_t_a_u[i-1]))) * delta_t - A_p_l / A_p_u * (math.exp(nom_omega_u/ 10) - 1) * delta_t_o
            if nom_delta_t_u < 0:
                nom_delta_t_u = 0
            nom_t_a_u[i] = nom_t_a_u[i-1] + nom_delta_t_u
            #<<<<<<<<<<<<<<<<<<<<<<<<Upper flange temperature>>>>>>>>>>>>>>>>>>>>>>>
            delta_t_u = 1/ (c_a_uf* p_a*V_i_u)   * (A_p_t * (alfa_c * (t_o2 - t_a_u[i-1]) + eps_m * eps_f * stefan * ((t_o2 + 273) ** 4 - (t_a_u[i-1] + 273) ** 4)) + A_p_l * 1 / (1 + omega_u/3) * (lamda_p / d_p * (t_o2 - t_a_u[i-1]))-(3.5*10**-3*d+2.5)*wt*nom_lamda_a_tj/(d-t_f)*2*(nom_t_a_u[i-1]-nom_t_a_l[i-1])) * delta_t - A_p_l / A_p_u * (math.exp(omega_u/ 10) - 1) * delta_t_o
            if delta_t_u < 0:
                delta_t_u = 0
            t_a_u[i] = t_a_u[i-1] + delta_t_u

    return t_a_l,t_a_u,time
def iteration_algorithm_ic(d,t_f,d_p,T_max,T_limit,TempGap,alfa):
# Discarding the second-last temperature gap
    TempGap.pop(0)
    round(T_max,2)
    a = T_max-T_limit
# Appending the last temperature gap
    TempGap.append(a)
# Preventing jumping repeatly over the target
    if TempGap [0]* TempGap [1] < 0:
        alfa=alfa/2
# Increasing converging speed when the gap is small
    elif TempGap [0]* TempGap [1] > 0:
        alfa=alfa*1.5
# Modifying fire protection thickness based on the gap of temperature
    d_p=d_p-alfa*((T_max-T_limit)/min(T_max,T_limit)*d_p)
    # d_p=round(d_p,2)
    if d_p <= 0:
        d_p=0.1e-5
    # elif d_p >= 0.5*(d-2*t_f):
    #     d_p= 0.5*(d-2*t_f)-0.01
    return d_p,alfa,TempGap
def iteration_algorithm(d,t_f,d_p,T_max,T_limit,TempGap,alfa):
# Discarding the second-last temperature gap
    TempGap.pop(0)
    round(T_max,2)
    a = T_max-T_limit
# Appending the last temperature gap
    TempGap.append(a)
# Preventing jumping repeatly over the target
    if TempGap [0]* TempGap [1] < 0:
        alfa=alfa/2
# Increasing converging speed when the gap is small
    elif TempGap [0]* TempGap [1] > 0:
        alfa=alfa*1.5
# Modifying fire protection thickness based on the gap of temperature
    d_p=alfa*((T_max-T_limit)/min(T_max,T_limit)*d_p)+d_p
    # d_p=round(d_p,2)
    if d_p <= 0:
        d_p=0.1
    elif d_p >= 0.5*(d-2*t_f):
        d_p= 0.5*(d-2*t_f)-0.01
    return d_p,alfa,TempGap

def insu_thickness_cal(time,tgt_temp,temp_allwce,d,w,t_f,t_w,lamda_p):
    # time in s
    time=time*60
    time=np.linspace(0,time,int(time/50))
    time=list(float(x) for x in time)
    nom_t_a_l=[0]*len(time)
    nom_t_a_u=[0]*len(time)
    t_a_l=[0]*len(time) 
    t_a_u=[0]*len(time)
    t_a_w=[0]*len(time)
    lamda_p=10**-3*lamda_p
    # t_o=[20,400.042,511.363,573.276,613.413,648.226,676.455,698.798,719.587,736.827,750.978,765.204,779.789,791.65,800.858,813.575,820.151,830.085,840.383,847.181,854.196,861.373,868.239,875.008,881.498,887.727,893.69,901.069,905.417,910.185,915.343,920.333,925.153,929.828,934.366,938.762,943.038,947.198,951.239,955.178,959.019,962.757,966.409,969.975,973.454,976.857,980.186,983.438,986.624,989.745,992.799,995.794,998.732,1001.61,1004.43,1007.21,1009.93,1012.6,1015.23,1017.81,1020.35,1022.85,1025.3,1027.72,1030.1,1032.44,1034.74,1037.01,1039.24,1041.45,1043.62,1045.76,1048.76]
    # time_o=[0,97.2868,198.242,298.792,392.456,495.336,598.215,695.534,799.387,896.919,986.423,1084.65,1195.85,1294.83,1376.8,1498.92,1566.36,1673.86,1793.07,1876.45,1966.41,2062.97,2159.53,2259.53,2359.53,2459.53,2559.53,2688.63,2767.73,2857.29,2957.29,3057.29,3157.29,3257.29,3357.29,3457.29,3557.29,3657.29,3757.29,3857.29,3957.29,4057.29,4157.29,4257.29,4357.29,4457.29,4557.29,4657.29,4757.29,4857.29,4957.29,5057.29,5157.29,5257.29,5357.29,5457.29,5557.29,5657.29,5757.29,5857.29,5957.29,6057.29,6157.29,6257.29,6357.29,6457.29,6557.29,6657.29,6757.29,6857.29,6957.29,7057.29,7200]
    t_o=[20,858.7446547,937.3228412,989.2593147,1024.61115,1048.677581,1092.495753,1098.902746,1099.839562,1099.976541,1099.99657,1099.999498,1099.999927,1099.999989,1099.999998,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100]
    time_o=[0,60,120,180,240,300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600,3900,4200,4500,4800,5100,5400,5700,6000,6300,6600,6900,7200]    
    lamda_a=[0.05333,0.052,0.05067,0.04933,0.048,0.04667,0.04534,0.04401,0.04267,0.04134,0.04,0.03868,0.03734,0.03601,0.034678,0.03401,0.03252,0.03102,0.02952,0.02835,0.0273,0.0273,0.0273,0.0273,0.0273]
    t_lamda=[20,60,100,140,180,220,260,300,340,380,420,460,500,540,580,600,645,690,735,770,805,840,875,900,1200]
    c_a=[439.8,465.77,487.62,506.18,522.33,536.9,550.75,564.74,579.71,596.51,616.01,639.06,666.5,699.18,737,97,760.21,805.8,936.87,5000,1001.92,785.81,708.48,668.75,650,650]
    t_c=[20,60,100,140,180,220,260,300,340,380,420,460,5000,540,580,6000,645,690,735,770,805,840,875,900,1200]
    A_p_b = (w + t_f) * 2
    V_i_l = w * t_f
    A_p_t = w
    A_p_l = w + 2 * t_f
    A_p_u = (w + t_f) * 2
    V_i_u = w * t_f
    A_p_w = (d - 2 * t_f) * 2
    V_i_w = (d - 2 * t_f) * t_w
    d_web = (d - t_f)
    stefan=5.67e-14
    alfa_c=50e-5
    eps_m=0.8
    eps_f=1
    p_p = 0.0000024
    c_p = 1047
    p_a = 7.9e-6
    # Stacking the temperature gaps of the last two cycles
    TempGap=[0,0]
    d_p=40
    alfa=1.01
    for j in range(100):
        for i in range(len(time)):
            print('time:',time[i])
            #<<<<<<<<<<<<<<< Calcualtion of first iteration >>>>>>>>>>>>>>>>>>
            if i ==0:
                #<<<<<<<<<<<<<<< Interpolation of the fire temperature >>>>>>>>>>>>>>>>>>>>>>>>>>
                for j in range(len(time_o)-1):
                    if time[i] >= time_o[j] and time_o[j+1] > time[i]:
                        t_o2 = (time[i] - time_o[j]) * (t_o[j+1] - t_o[j]) / (time_o[j+1] - time_o[j]) + t_o[j]
                        break
                #Ambient temperature
                delta_t_o=t_o2
                #Simulation time
                delta_t=time[i]
                #<<<<<<<<<<<<<<< Interpolation of the upper flange nonminal specific heat  >>>>>>>>>>>>>>>>>>>>>>>>>>
                for j in range(24):
                    if 20 >= t_c[j] and t_c[j+1] > 20:
                        nom_c_a_uf = (20 - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                        break
                #Interpolation for the top junction steel conductivity
                for j in range(24):
                    if 20 >= t_lamda[j] and t_lamda[j+1] > 20:
                        nom_lamda_a_tj = (20- t_lamda[j]) * (lamda_a[j+1] - lamda_a[j]) / (t_lamda[j+1] - t_lamda[j]) + lamda_a[j]
                        break
                #<<<<<<<<<<<<<<<<<<<<<<<<   Interpolation of the upper flange specifica heat   >>>>>>>>>>>>>>>>>>>>>>>>>>
                for j in range(24):
                    if 20 >= t_c[j] and t_c[j+1] > 20:
                        c_a_uf = (20 - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                        break
                #<<<<<<<<<<<<<<<  Interpolation of the lower flange nominal specific heat  >>>>>>>>>>>>>>>>>>>>>>>>>>
                for j in range(24):
                    if 20 >= t_c[j] and t_c[j+1] > 20:
                        nom_c_a_lf = (20 - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                        break
                #Interpolation for the bottom junction steel conductivity
                for j in range(24):
                    if 20 >= t_lamda[j] and t_lamda[j+1] > 20:
                        nom_lamda_a_bj = (20 - t_lamda[j]) * (lamda_a[j+1] - lamda_a[j]) / (t_lamda[j+1] - t_lamda[j]) + lamda_a[j]
                        break 
                #<<<<<<<<<<<<<<<<<<<<<<<<   Interpolation of the lower flange specifica heat   >>>>>>>>>>>>>>>>>>>>>>>>>>
                for j in range(24):
                    if 20 >= t_c[j] and t_c[j+1] > 20:
                        c_a_lf = (20 - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                        break
                #<<<<<<<<<<<<<<<<<<<<<<<<   END   >>>>>>>>>>>>>>>>>>>>>>>>>>

                nom_omega_l = (c_p*p_p/nom_c_a_lf/p_a)*d_p*(A_p_b/V_i_l)
                nom_omega_u = (c_p*p_p/nom_c_a_uf/p_a)*d_p*(A_p_u/V_i_u)
                omega_l = (c_p*p_p/c_a_lf/p_a)*d_p*(A_p_b/V_i_l)
                omega_u = (c_p*p_p/c_a_uf/p_a)*d_p*(A_p_u/V_i_u)
                #<<<<<<<<<<<<<<<<<<<<<<<<  Nominal lower flange temperature  >>>>>>>>>>>>>>>>>>>>>
                nom_delta_t_l = lamda_p/d_p/ (nom_c_a_lf* p_a*V_i_l)   * (A_p_b * 1 / (1 + nom_omega_l/ 3) * (t_o2 - nom_t_a_l[i-1])) * delta_t - (math.exp(nom_omega_l/ 10) - 1) * delta_t_o
                if nom_delta_t_l < 0:
                    nom_delta_t_l = 0
                nom_t_a_l[i] =20 + nom_delta_t_l
            #<<<<<<<<<<<<<<<<<<  Calculation of the rest iterations  >>>>>>>>>>>>>>>>>>>>>>>>>        
            elif i != 0:
                #<<<<<<<<<<<<<<< Interpolation of the temperature >>>>>>>>>>>>>>>>>>>>>>>>>>
                for j in range(len(time_o)-1):
                    if time[i-1] >= time_o[j] and time_o[j+1] > time[i-1]:
                        t_o1 = (time[i-1] - time_o[j]) * (t_o[j+1] - t_o[j]) / (time_o[j+1] - time_o[j]) + t_o[j]
                        break
                for j in range(len(time_o)-1):
                    if time[i] >= time_o[j] and time_o[j+1] > time[i]:
                        t_o2 = (time[i] - time_o[j]) * (t_o[j+1] - t_o[j]) / (time_o[j+1] - time_o[j]) + t_o[j]
                        break
                #Temperature increment
                delta_t_o=t_o2-t_o1
                #Time increment
                delta_t=time[i]-time[i-1]
                #Interpolation for the top junction steel conductivity
                for j in range(24):
                    if nom_t_a_u[i-1] >= t_lamda[j] and t_lamda[j+1] > nom_t_a_u[i-1]:
                        nom_lamda_a_tj = (nom_t_a_u[i-1]- t_lamda[j]) * (lamda_a[j+1] - lamda_a[j]) / (t_lamda[j+1] - t_lamda[j]) + lamda_a[j]
                        break
                #Interpolation for the nominal upper flange specific heat
                for j in range(24):
                    if nom_t_a_u[i-1] >= t_c[j] and t_c[j+1] > nom_t_a_u[i-1]:
                        nom_c_a_uf = (nom_t_a_u[i-1] - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                        break
                #Interpolation for the upper flange specific heat    
                for j in range(24):
                    if t_a_u[i-1] >= t_c[j] and t_c[j+1] > t_a_u[i-1]:
                        c_a_uf = (t_a_u[i-1] - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                        break
                #Interpolation for the bottom junction steel conductivity
                for j in range(24):
                    if nom_t_a_l[i-1] >= t_lamda[j] and t_lamda[j+1] > nom_t_a_l[i-1]:
                        nom_lamda_a_bj = (nom_t_a_l[i-1] - t_lamda[j]) * (lamda_a[j+1] - lamda_a[j]) / (t_lamda[j+1] - t_lamda[j]) + lamda_a[j]
                        break          
                print('lamda_a_bj:',nom_lamda_a_bj)
                #Interpolation for the nominal lower flange specific heat
                for j in range(24):
                    if nom_t_a_l[i-1] >= t_c[j] and t_c[j+1] > nom_t_a_l[i-1]:
                        nom_c_a_lf = (nom_t_a_l[i-1] - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                        break
                #Interpolation for the lower flange specific heat    
                for j in range(24):
                    if t_a_l[i-1] >= t_c[j] and t_c[j+1] > t_a_l[i-1]:
                        c_a_lf = (t_a_l[i-1] - t_c[j]) * (c_a[j+1] - c_a[j]) / (t_c[j+1] - t_c[j]) + c_a[j]
                        break
                #<<<<<<<<<<<<<<<<<<<<<<<<   END   >>>>>>>>>>>>>>>>>>>>>>>>>>u
                nom_omega_l = (c_p*p_p/nom_c_a_lf/p_a)*d_p*(A_p_b/V_i_l)
                nom_omega_u = (c_p*p_p/nom_c_a_uf/p_a)*d_p*(A_p_u/V_i_u)
                omega_l = (c_p*p_p/c_a_lf/p_a)*d_p*(A_p_b/V_i_l)
                omega_u = (c_p*p_p/c_a_uf/p_a)*d_p*(A_p_u/V_i_u)            
                #<<<<<<<<<<<<<<<<<<<<<<<<  Nominal lower flange temperature  >>>>>>>>>>>>>>>>>>>>>
                nom_delta_t_l = lamda_p/d_p/ (nom_c_a_lf* p_a*V_i_l)   * (A_p_b * 1 / (1 + nom_omega_l/ 3) * (t_o2 - nom_t_a_l[i-1])) * delta_t - (math.exp(nom_omega_l/ 10) - 1) * delta_t_o
                if nom_delta_t_l < 0:
                    nom_delta_t_l = 0
                nom_t_a_l[i] = nom_t_a_l[i-1] + nom_delta_t_l    
        print('nom_t_a_l:',nom_t_a_l)    
        if nom_t_a_l[-1] > tgt_temp or nom_t_a_l[-1] <= tgt_temp - temp_allwce:
            d_p,alfa,TempGap=iteration_algorithm(d,t_f,d_p,nom_t_a_l[-1],tgt_temp,TempGap,alfa)
        else:
            print('temp_lower flange:',nom_t_a_l[-1])
            break
    return d_p,time
time_c_arry=[900,3600,7200]
HC_list=[120]
tgt_temp_list=[400]
dep=406
wid=280
ft=20
wt=10
# insu,time=insu_thickness_cal(60,400,5,400,180,13.5,8.6,0.02)
# print('sinulation conducticity',str(insu))

for k in range(len(HC_list)):
    for j in range(len(tgt_temp_list)):
        time_p=HC_list[k]
        HC=HC_list[k]
        tgt_temp=tgt_temp_list[j]
        header=[str(dep)+'x'+str(wid)+'x'+str(ft)+'x'+str(wt)]
        csv.writer(open('coor_temp.csv','a',newline='')).writerow(header)
        header=['HC'+str(HC)+' '+str(tgt_temp)+'°C']
        csv.writer(open('coor_temp.csv','a',newline='')).writerow(header)
        header=[str(int(time_c_arry[0]/60))+' min','',str(int(time_c_arry[1]/60))+' min','',str(int(time_c_arry[2]/60))+' min','']
        csv.writer(open('coor_temp.csv','a',newline='')).writerow(header)
        header=['Vertical coordinate (mm)','Temperature (°C)','Vertical coordinate (mm)','Temperature (°C)','Vertical coordinate (mm)','Temperature (°C)','Vertical coordinate (mm)','Temperature (°C)','Vertical coordinate (mm)','Temperature (°C)']
        horizontal=[[],[],[],[],[]]
        for i in range(len(time_c_arry)):
            # Protection standard
            temp_allwce=5
            lamda_p=2*10**-1
            # Section dimensions
            # Cal_time
            yc=0
            time_c=time_c_arry[i]
            insu,time=insu_thickness_cal(time_p,tgt_temp,temp_allwce,dep,wid,ft,wt,lamda_p)
            sec_fac=(2*dep+3*wid-2*wt)/(2*ft*wid+(dep-2*ft)*wt)
            if insu >= (dep-2*ft)/2:
                insu=(dep-2*ft)/2-0.0001
                print('Insulation thickness Overbound')
            t_a_l,t_a_u,time_c=temp_cal(time_c,dep,wid,ft,wt,insu,lamda_p)
            t_a_l_f= t_a_l[-1]
            t_a_u_f= t_a_u[-1]
            print('time_c:',time_c)
            time= time_c[-1]
            if HC==60:
            # HC60 400C
                if time ==900 and tgt_temp==400:
                    yc=1*dep+2*sec_fac - 95
                elif time ==1800 and tgt_temp==400:
                    yc=1*dep+2*sec_fac - 150
                elif time ==3600 and tgt_temp==400:
                    yc=1*dep+2*sec_fac - 300
                elif time==5400 and tgt_temp==400:
                    yc=0.9*dep+2*sec_fac - 300
                elif time==7200 and tgt_temp==400:
                    yc=600/((1+((dep-200)/450)**-2)*(1+((sec_fac-0.02)/0.03)**-2))
                # HC60 538C
                if time ==900 and tgt_temp==538:
                    yc=0.9*dep+2*sec_fac - 200
                elif time ==1800 and tgt_temp==538:
                    yc=0.8*dep+2*sec_fac - 120
                elif time ==3600 and tgt_temp==538:
                    yc=1*dep+2*sec_fac - 250
                elif time==5400 and tgt_temp==538:
                    yc=0.42*dep+2*sec_fac - 120
                elif time==7200 and tgt_temp==538:
                    yc=dep+2*sec_fac - 400
            elif HC==15:
            # HC15 400C
                if time ==900 and tgt_temp==400:
                    yc=1*dep+2*sec_fac - 100
                elif time ==1800 and tgt_temp==400:
                    yc=1.1*dep+sec_fac - 150
                elif time ==3600 and tgt_temp==400:
                    yc=1*dep+2*sec_fac
                elif time==5400 and tgt_temp==400:
                    yc=0.9*dep+2*sec_fac - 25
                elif time==7200 and tgt_temp==400:
                    yc=dep+2*sec_fac - 20
                # HC15 538C
                if time ==900 and tgt_temp==538:
                    yc=dep+2*sec_fac - 20
                elif time ==1800 and tgt_temp==538:
                    yc=0.95*dep+2*sec_fac
                elif time ==3600 and tgt_temp==538:
                    yc=1*dep+2*sec_fac
                elif time==5400 and tgt_temp==538:
                    yc=0.95*dep+2*sec_fac
                elif time==7200 and tgt_temp==538:
                    yc=dep+2*sec_fac - 20
            elif HC==120:
            # HC120 400C
                if time ==900 and tgt_temp==400:
                    yc=1*dep+2*sec_fac - 80
                elif time ==1800 and tgt_temp==400:
                    yc=1*dep+2*sec_fac - 140
                elif time ==3600 and tgt_temp==400:
                    yc=1*dep+2*sec_fac-250
                elif time==5400 and tgt_temp==400:
                    yc=0.9*dep+2*sec_fac - 250
                elif time==7200 and tgt_temp==400:
                    yc=dep+2*sec_fac - 400
                # HC120 538C
                if time ==900 and tgt_temp==538:
                    yc=0.7*dep+2*sec_fac - 180
                elif time ==1800 and tgt_temp==538:
                    yc=dep+2*sec_fac-150
                elif time ==3600 and tgt_temp==538:
                    yc=0.9*dep+2*sec_fac-180
                elif time==5400 and tgt_temp==538:
                    yc=0.9*dep+2*sec_fac-200
                elif time==7200 and tgt_temp==538:
                    yc=0.85*dep+2*sec_fac - 160
            if yc<=0:
                temp_dist=[[t_a_u_f,0],[t_a_u_f,(ft/dep)/2],[t_a_l_f,(ft/dep)/2+0.2],[t_a_l_f,1-(ft/dep)/2],[t_a_l_f,1]]
                temp=[t_a_u_f,t_a_u_f,t_a_l_f,t_a_l_f,t_a_l_f]
                coor=[dep,dep-ft/2,dep-ft/2-0.2*dep,ft/2,0]
            else:
                coor_mid=1-yc/dep
                coor_mid=round(coor_mid,8)
                temp_dist=[[t_a_u_f,0],[t_a_u_f,(ft/dep)/2],[t_a_l_f,coor_mid],[t_a_l_f,1-(ft/dep)/2],[t_a_l_f,1]]
                temp=[t_a_u_f,t_a_u_f,t_a_l_f,t_a_l_f,t_a_l_f]
                coor=[dep,dep-ft/2,dep-coor_mid*dep,ft/2,0]
            print(temp_dist,insu)
            color_list=['red','black','lime','blue','purple']
            line_style_list=['-','--','-.',':','-']
            plt.plot(temp,coor,line_style_list[i],color=color_list[i],label=f'Time={time}')
            plt.xlabel('Temperature (°C)',fontsize=12,fontstyle='italic',fontweight='bold')
            plt.ylabel('Height (mm)',fontsize=12,fontstyle='italic',fontweight='bold')
            plt.xticks(font='Times new roman',fontweight='bold',size=12)
            plt.yticks(font='Times new roman',fontweight='bold',size=12)
            plt.xlim(0,1100)
            plt.ylim(0,dep)

            for l in range(len(temp)):
                print(horizontal)
                horizontal[l].append(coor[l])
                horizontal[l].append(temp[l])
        for l in range(len(horizontal)):
            csv.writer(open('coor_temp.csv','a',newline='')).writerow(horizontal[l])
        plt.legend(loc='upper right',fontsize=12)
        plt.savefig('temp_dist'+str(HC)+str(tgt_temp)+'.tiff',dpi=400)
        plt.clf()