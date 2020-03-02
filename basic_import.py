import scipy.linalg as sla
import numpy.linalg as nla
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from scipy.linalg import cholesky, cho_solve, solve_triangular
import io
import dill
import _pickle as pickle
import os
import math
import functools
import contextlib
import re
import sys
from scipy.spatial import minkowski_distance
import glob
import numpy as np
from pathlib import Path
from IPython.lib.pretty import pretty as pretty_repr
from numba import jit,vectorize, float64
from tqdm import tqdm
import zipfile
from recordclass import recordclass
GPR = recordclass('GPR', ('kernel', 'X_train', 'alpha','K_inv'))
GPR_list = recordclass('GPR_list', ('m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13', 'm14', 'm15', 'm16', 'm17', 'm18', 'm19', 'm20'))

from numpy import \
    pi, nan, \
    log, log10, exp, sin, cos, tan, \
    sqrt, floor, ceil, \
    mean, prod, \
    diff, cumsum, \
    array, asarray, asmatrix, \
    linspace, arange, reshape, \
    eye, zeros, ones, diag, trace
from numpy.random import rand,randn
from numpy.linalg import eig
from scipy.linalg import svd, sqrtm, inv, eigh
from numpy import linalg


from localization import partial_direct_obs_nd_loc_setup as loc_setup

def generate_set_DA(T = 10000,gpr_alpha2=1,noise_std=0.5,gpr_alpha=0.3,gpr_factor=0.4,change_seed2get_obs=False,truth_new=False,obs_new_sigma=False,fau = None,obs_new_sigma_value=None,std_infl = 1,infl_not =True,gamma_test =True,adpative_infl = False,infl = 1.02,metrics_list = None,da_method = 'LETKF',spinup = 999,updateInterval = 1,dt = 0.05, a1=0.55,a2=0.55,data_path = None,pbar = False,obs_type='linear', save_ens_or_not=False,data_saved_file = None,ens_size = 100,local_scale=1):
    J = 40  # dimension of Lorenz-96 Model
    commonStartState = np.zeros(J)  # start-vector
    commonStartState[5] = 0.01
    dt = dt
    T = T
    spinup = spinup
    updateInterval = updateInterval
    obs_inds = np.arange(0, J, 2)  # only half of the model states are observed
    obs_num = len(obs_inds)
    ens_size = ens_size
    local_scale=local_scale
    da_method=da_method
    save_ens_or_not=save_ens_or_not
    pbar = pbar
    infl_not = infl_not
    infl = infl
    adpative_infl = adpative_infl
    gamma_test = gamma_test
    std_infl = std_infl
    truth_new=truth_new
    change_seed2get_obs=change_seed2get_obs
    gpr_alpha = gpr_alpha
    gpr_factor = gpr_factor
    noise_std=noise_std
    gpr_alpha2 = gpr_alpha2

    obs_new_sigma=obs_new_sigma
    if fau is None:
        fau = ['f','a','u']
    if obs_new_sigma :
        if obs_new_sigma_value is not None:
            obs_new_sigma_value = obs_new_sigma_value
        else:
            print('check the obs_new_sigma_value')

    if metrics_list is None:
        metrics_list =  ['mu', 'var', 'err', 'rh', 'rmse', 'rmv','normaltest','logp_m','skew','kurt','mad','x1','x2']

    set_DA = {'J': J,  # (Model paramter)  the dimension of model states

              # related to time settings
              'dt': dt,
              'T': T,  # (DA parameter)    the time step for DA
              # (DA parameter)    the number of particles or the size of the ensemble
              'N': ens_size,
              'spinup': spinup,  # (DA parameter)    time that the ensemble is run before data assimilation starts to
              'updateInterval': updateInterval,  # (DA parameter)   how often is the ensemble updated with observations

              'update_times': len(np.arange(spinup + updateInterval, T + 1, updateInterval)),
              'update_time_inds': np.arange(spinup + updateInterval, T + 1, updateInterval),
              'update_t': np.arange(0, len(np.arange(spinup + updateInterval, T + 1, updateInterval)), 1),

              'da_start_time': spinup + 1,
              'da_time_len': T - spinup,
              'da_t': np.arange(0, T - spinup, 1),
              'da_t_inds': np.arange(spinup + 1, T + 1, 1),

              'local_scale': local_scale,  # (DA localization)
              'local_dic': 'x2y',  # (DA localization)
              'local_func': 'GC',  # (DA localization)

              'infl':infl,
              'adaptive_infl':adpative_infl,
              'infl_not':infl_not,
              'std_infl':std_infl,

              'obs_sigma': 1,  # (Obervation parameter) the std of observations
              'obs_ens': True,  # (DA parameter)  to decide to use Observation ensemble or not
              'obs_num': obs_num,  # (Obervation parameter) the number of obervations
              'obs_inds': obs_inds.tolist(),  # (Obervation parameter) the index of obervations
              'obs_type': obs_type,  # linear
              'obs_new_sigma':obs_new_sigma,
              'obs_new_sigma_value': obs_new_sigma_value,

              'data_saved_path': data_path,
              'data_saved_file_name': data_saved_file,
              'data_input_path': 'input_data',
              'settings_path': 'settings',
              'da_methods': da_method,
              'save_ens_or_not':save_ens_or_not,
              'pbar':pbar,
              'a1': a1,
              'a2': a2,
              'metrics_list':metrics_list,
              'gamma_test':gamma_test,
              'fau':fau,

              'truth_new':truth_new,
              'change_seed2get_obs':change_seed2get_obs,

              'gpr_alpha':gpr_alpha,
              'gpr_factor':gpr_factor,
              'noise_std':noise_std,
              'gpr_alpha2':gpr_alpha2

              }

    return set_DA

def generare_metrics(set_DA):

    metrics_list = set_DA['metrics_list']
    # metrics_list = ['mu', 'var', 'err', 'rh', 'rmse', 'rmv','normaltest','logp_m','skew','kurt','mad']
    dict = {}

    if 'mu' in metrics_list:
        dict_tmp = {'mu': FAU_series(set_DA, set_DA['J'])}
        dict.update(dict_tmp)
    if 'var' in metrics_list:
        dict_tmp = {'var': FAU_series(set_DA, set_DA['J'])}
        dict.update(dict_tmp)
    if 'mad' in metrics_list:
        dict_tmp = {'mad': FAU_series(set_DA, set_DA['J'])}
        dict.update(dict_tmp)
    if 'err' in metrics_list:
        dict_tmp = {'err': FAU_series(set_DA, set_DA['J'])}
        dict.update(dict_tmp)
    if 'rh' in metrics_list:
        dict_tmp = {'rh': FAU_series(set_DA, set_DA['J'])}
        dict.update(dict_tmp)
    if 'normaltest' in metrics_list:
        dict_tmp = {'normaltest': FAU_series(set_DA, set_DA['J'])}
        dict.update(dict_tmp)
    if 'svals' in metrics_list:
        dict_tmp = {'svals': FAU_series(set_DA, min(set_DA['J'], set_DA['N']))}
        dict.update(dict_tmp)
    if 'umisf' in metrics_list:
        dict_tmp = {'umisf': FAU_series(set_DA, min(set_DA['J'], set_DA['N']))}
        dict.update(dict_tmp)
    if 'logp_m' in metrics_list:
        dict_tmp = {'logp_m': FAU_series(set_DA, 1)}
        dict.update(dict_tmp)
    if 'skew' in metrics_list:
        dict_tmp = {'skew': FAU_series(set_DA, 1)}
        dict.update(dict_tmp)
    if 'rmv' in metrics_list:
        dict_tmp = {'rmv': FAU_series(set_DA, 1)}
        dict.update(dict_tmp)
    if 'kurt' in metrics_list:
        dict_tmp = {'kurt': FAU_series(set_DA, 1)}
        dict.update(dict_tmp)
    if 'rmse' in metrics_list:
        dict_tmp = {'rmse': FAU_series(set_DA, 1)}
        dict.update(dict_tmp)

    if 'x1' in metrics_list:
        dict_tmp = {'x1': FAU_series(set_DA, set_DA['N'])}
        dict.update(dict_tmp)
    if 'x2' in metrics_list:
        dict_tmp = {'x2': FAU_series(set_DA, set_DA['N'])}
        dict.update(dict_tmp)
    return dict

def generate_settings(set_DA):
    J = set_DA['J']  # dimension of Lorenz-96 Model

    dt = set_DA['dt']
    T = set_DA['T']
    commonStartState = np.zeros(J)  # start-vector
    commonStartState[5] = 0.01


    # settings data in dict for export to YAML file
    settings = {
        'J': J,  # (Model paramter)  the dimension of model states
        'F': 8.0,  # (Model parameter) to set the choas in the Lorenz model
        'dt': dt,  # (Model parameter) the time step for Lorenz model
        # (Model parameter) the initial ensemble
        'startState': commonStartState.tolist(),

        'startTime': 0.0,  # We don't need start time
        'endTime': T * dt,  # We don't nedd end time
    }

    return settings


def H_nonlinear_design_1(x):
    # 9_21
    return np.log(abs(x))*0.2-np.exp(x/10)*0.01  + 1.5*x - 2.5*np.sqrt(abs(x))+np.square(x/5)*0.2

def H_operator(ens,set_DA):
    # the "observation operator" that links model-space to observation-space

    # input Nx: the dimension of model states
    # obs_inds: the index of obervations out of the index of Nx
    prm = container(set_DA)
    obs_inds = prm.obs_inds
    Nx = prm.J


    Ny = len(obs_inds)
    H = np.zeros((Ny, Nx))
    H[range(Ny), obs_inds] = 1

    if prm.obs_type == 'linear':
        pass
    elif prm.obs_type == 'ln':
        ens = np.log(abs(ens))
    elif prm.obs_type == 'abs':
        ens = abs(ens)
    elif prm.obs_type =='non':
        ens = H_nonlinear_design_1(ens)

    return ens@H.T

@jit
def Lorenz96_fun(x,t,m=40):
    force =8
    # compute state derivatives
    d = np.zeros(m)
    # first the 3 edge cases: i=1,2,N
    d[0] = (x[1] - x[m - 2]) * x[m - 1] - x[0]
    d[1] = (x[2] - x[m - 1]) * x[0] - x[1]
    d[m - 1] = (x[0] - x[m - 3]) * x[m - 2] - x[m - 1]
    # then the general case
    for i in range(2, m - 1):
        d[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]
    # add the forcing term
    d = d + force

    # return the state derivatives
    return d

@jit
def l96_fun_prm(x0, set_DA):
    x = x0
    t = 2
    Nx= set_DA['J']
    dt =0.05 # set_DA['dt']

    """Runge-Kutta N-th order (explicit, non-adaptive) numerical ODE solvers."""
    # k1 = dt * f(t, x)
    # k2 = dt * f(t + dt / 2, x + k1 / 2)
    # k3 = dt * f(t + dt / 2, x + k2 / 2)
    # k4 = dt * f(t + dt, x + k3)

    k1 = dt * Lorenz96_fun(x, t,m=Nx)
    k2 = dt * Lorenz96_fun(x + k1 / 2, t + dt / 2,m=Nx)
    k3 = dt * Lorenz96_fun(x + k2 / 2, t + dt / 2,m=Nx)
    k4 = dt * Lorenz96_fun(x + k3, t + dt,m=Nx)
    updated_x = x + (k1 + 2 * (k2 + k3) + k4) / 6

    return updated_x


# @jit
# def Lorenz96_fun(x,t,force = 8.0,m=40):
#     # compute state derivatives
#     d = np.zeros(m)
#     # first the 3 edge cases: i=1,2,N
#     d[0] = (x[1] - x[m - 2]) * x[m - 1] - x[0]
#     d[1] = (x[2] - x[m - 1]) * x[0] - x[1]
#     d[m - 1] = (x[0] - x[m - 3]) * x[m - 2] - x[m - 1]
#     # then the general case
#     for i in range(2, m - 1):
#         d[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]
#     # add the forcing term
#     d = d + force
#
#     # return the state derivatives
#     return d
#
# @jit
# def l96_fun_prm(x0,dt=0.05):
#     x = x0
#     t = 2
#     force = 8.0
#
#     """Runge-Kutta N-th order (explicit, non-adaptive) numerical ODE solvers."""
#     # k1 = dt * f(t, x)
#     # k2 = dt * f(t + dt / 2, x + k1 / 2)
#     # k3 = dt * f(t + dt / 2, x + k2 / 2)
#     # k4 = dt * f(t + dt, x + k3)
#
#     k1 = dt * Lorenz96_fun(x, t)
#     k2 = dt * Lorenz96_fun(x + k1 / 2, t + dt / 2)
#     k3 = dt * Lorenz96_fun(x + k2 / 2, t + dt / 2)
#     k4 = dt * Lorenz96_fun(x + k3, t + dt)
#     updated_x = x + (k1 + 2 * (k2 + k3) + k4) / 6
#
#     return updated_x

def get_new_truth(set_DA):

    truth_all = np.zeros((15000,set_DA['J']))
    truth_all[0, :] = np.random.random(40)

    for loop in range(1,15000):
        truth_all[loop, :] = l96_fun_prm(truth_all[(loop-1), :])  # l96(ens_total_tmp[loop, :].ravel()).rk4()

    return truth_all

def change_std(truth,set_DA):
    change_seed = set_DA['change_seed2get_obs']
    new_std = set_DA['obs_new_sigma_value']

    if change_seed:
        pass
    else:
        np.random.seed(924)
    ens_size,state = truth.shape

    ens = np.random.normal(0, new_std, truth.shape)
    ens_mean = np.tile(ens.mean(axis=1), [state, 1])
    new_noise = ens - ens_mean.T
    return new_noise + truth

def get_obs_for_DA(obs, set_DA):
    # to generate Observation ensemble or not
    tmp = np.random.randn(len(obs))
    return obs + set_DA['obs_sigma'] * (tmp - tmp.mean())

# jit functions
from numba import jit,vectorize,float64

@jit(nopython=True)
def weight_jit(ob_loc, HE_loc,obs_std = 1, sacle=1):
    ob_var = obs_std * sacle
    error = ob_loc - HE_loc
    w_tmp = np.exp(-np.square(error ) / (2 * (ob_var * ob_var) ** 2)) / (
                    math.sqrt(2 * math.pi) * ob_var)
    return w_tmp

@jit(nopython=True)
def weight_factor_jit(w_tmp,coeff, factor=1):
    w_tmp_after_mul = (np.multiply(w_tmp, coeff) + factor) * factor
    return w_tmp_after_mul




@jit(nopython=True)
def just_weight_simple_jit(ob_loc, HE_loc, coeff,ob_var = 1, factor=1, sacle=1):

    error = ob_loc - HE_loc
    w_tmp = np.exp(-np.square(error ) / (2 * (ob_var * ob_var) ** 2)) / (
                    math.sqrt(2 * math.pi) * ob_var * sacle)
    # coeff_h = np.tile(coeff[None, :], (N, 1))
    w_tmp_after_mul = (np.multiply(w_tmp, coeff) + factor) * factor

    return w_tmp_after_mul


def pf_obs_factor_GPR(set_DA,y_predict, x_predict, coeff=1):
    # obs = ob_loc
    # HE = HE_loc
    obs, HE = y_predict, x_predict
    N=set_DA['N']
    ob_var = set_DA['obs_sigma']
    std_infl = set_DA['std_infl']
    frac_neff = set_DA['a2']

    max_infl = 1  # ?????????????????????????????????????
    Neff = frac_neff * N

    w_ini = weight_jit(obs, HE, obs_std=ob_var, sacle=std_infl)  # ob_var
    w = weight_factor_jit(w_ini, coeff)
    w_ini_nor=w/sum(w)

    # w = just_weight_simple_jit(obs, HE, local_coeff)
    Neff_init = neff(w_ini_nor)

    if ((abs(Neff_init - Neff) > 1) or (sum(np.square(w_ini)) == 0.0)):
        ke = 1
        ks = 0.001
        tol = 0.00001

        # w_ini = weight_jit(obs, HE, obs_std=ob_var, sacle=1)
        fks = Neff - neff( weight_factor_jit(w_ini, coeff, factor=ks))
        fke = Neff - neff(weight_factor_jit(w_ini, coeff,factor=ke))
        if fks * fke>0:
            for infl in range(10):
                w_ini = weight_jit(obs, HE, obs_std=ob_var, sacle=std_infl+1+infl)
                fks = Neff - neff(weight_factor_jit(w_ini, coeff, factor=ks))
                fke = Neff - neff(weight_factor_jit(w_ini, coeff, factor=ke))
                if fks * fke < 0:
                    break
            # if fks * fke > 0:
            #     print(obs)
            #     print('fks * fke>0')

        for i in range(100):
            km = (ke + ks) / 2.0

            w = weight_factor_jit(w_ini, coeff,factor=ks)  # just_weight_simple_jit(obs, HE, local_coeff, factor=ks)
            fks = Neff - neff(w)

            w = weight_factor_jit(w_ini, coeff,
                                  factor=ke)  # just_weight_simple_jit(obs, HE, local_coeff, factor=ke)
            fke = Neff - neff(w)

            if i == 0:
                if fks < 0.0:
                    break
                if fke * fks >= 0.0:
                    ke = ke * 10.0
                    w = weight_factor_jit(w_ini, coeff,
                                          factor=ke)  # just_weight_simple_jit(obs, HE, local_coeff, factor=ke)
                    fke = Neff - neff(w)
                    km = (ke + ks) / 2.0
                    if (ke > 100000.0):
                        break
            # Solution likely impossible so use default value
            if (ke > 100000.0):
                km = 1.0
                break
            # Stop if ke is not a number
            if ke != ke:
                km = max_infl
                break
            # Evaluate function at mid points
            w = weight_factor_jit(w_ini, coeff,
                                  factor=km)  # just_weight_simple_jit(obs, HE, local_coeff, factor=km)
            fkm = Neff - neff(w)

            # Exit critera
            if ((ke - ks) / 2.0 < tol):
                break
            # New end points
            if (fkm * fks > 0.0):
                ks = km
            else:
                ke = km
        # Set inflation
        obs_err_factor = km

        w_final = weight_factor_jit(w_ini, coeff,
                                    factor=obs_err_factor)  # just_weight_simple_jit(obs, HE, local_coeff, factor=obs_err_factor)
        w_final_nor = w_final / sum(w_final)
        Neff_final = neff(w_final_nor)
        if ((abs(Neff - Neff_final) > 1.0) or (Neff_final != Neff_final)):
            obs_err_factor = 1000

        if obs_err_factor == km:
            w_final = w_final
        else:
            w_final = weight_factor_jit(w_ini, coeff,
                                        factor=obs_err_factor)  # just_weight_simple_jit(obs, HE, local_coeff, factor=obs_err_factor)
            w_final_nor = w_final / sum(w_final)

    else:
        obs_err_factor = 1000
        w_final = w_ini
        w_final_nor = w_ini_nor

    return obs_err_factor, w_final,w_final_nor

@vectorize([float64(float64, float64)])
def f(x,y):
    return x*y


@jit(nopython=True)
def neff(w):
    w = w / np.sum(w)
    return 1.0 / np.sum(np.square(w))


# to save data and to load data


def data_saved(data, set_DA, data_name=None, path=None):

    if path is None and ('set' not in data_name):
        path = os.path.join(set_DA['data_saved_path'], set_DA['data_saved_file_name'])
    elif path is None and ('set' in data_name):
        path = os.path.join(set_DA['data_saved_path'], set_DA['settings_path'])
    else:
        path = os.path.abspath(path)

    if os.path.exists(path):
        name = os.path.join(path, data_name)
        with open(os.path.abspath(name), 'wb') as dill_file:
            dill.dump(data, dill_file)
    else:
        os.mkdir(path)
        name = os.path.join(path, data_name)
        with open(os.path.abspath(name), 'wb') as dill_file:
            dill.dump(data, dill_file)

    # if path is None and ('set' not in data_name):
    #     path = Path(set_DA['data_saved_path']) / set_DA['data_saved_file_name']
    # elif path is None and ('set' in data_name):
    #     path = Path(set_DA['data_saved_path']) / set_DA['settings_path']
    # else:
    #     path = Path(path)
    #
    # if path.is_dir():
    #     # with (path / data_name).open('wb') as dill_file:
    #     #     dill.dump(data, dill_file)
    #
    #     # with open(path + '/' + data_name, 'wb') as dill_file:
    #     #     dill.dump(data, dill_file)
    #
    #     with open(os.path.abspath(path / data_name), 'wb') as dill_file:
    #         dill.dump(data, dill_file)
    #
    # else:
    #     path.mkdir()
    #     # with (path / data_name).open('wb') as dill_file:
    #     #     dill.dump(data, dill_file)
    #     # os.mkdir(path)
    #     # with open(path + '/' + data_name, 'wb') as dill_file:
    #     #     dill.dump(data, dill_file)
    #     with open(os.path.abspath(path / data_name), 'wb') as dill_file:
    #         dill.dump(data, dill_file)


def load_data(data_name, set_DA = None, path=None):

    if path is None and ('set' not in data_name):
        path = Path(set_DA['data_saved_path']) / set_DA['data_input_path']
    elif path is None and ('set' in data_name):
        path = Path(set_DA['data_saved_path']) / set_DA['settings_path']
    else:
        path = Path(path)

    if set_DA is not None:
        if set_DA['dt'] == 0.05:
            dt = '0_05'

    if data_name == 'ens_ini':
        # data_name_pickle = path + '/ens_' + str(set_DA['N']) + '.pickle'
        # pkl_file = open(data_name_pickle, 'rb')
        # ens = pickle.load(pkl_file)
        # pkl_file.close()

        if set_DA['J'] == 40:
            with (path / ('ens_' + str(set_DA['N']) + '.pickle')).open('rb') as pkl_file:
                ens = pickle.load(pkl_file)
        else:
            with (path / ('ens_' + str(set_DA['J']) + '_' + str(set_DA['N']) +  '.pickle')).open('rb') as pkl_file:
                ens = pickle.load(pkl_file)

        return ens

    elif data_name == 'truth':
        # file_name = path + '/X_true_' + dt + '.pickle'
        # pkl_file = open(file_name, 'rb')
        # truth_all = pickle.load(pkl_file)
        # pkl_file.close()
        if set_DA['J'] == 40:
            with (path / ('X_true_' + dt + '.pickle')).open('rb') as pkl_file:
                truth_all = pickle.load(pkl_file)
        else:
            with (path / ('truth_' + str(set_DA['J']) + '_' + str(set_DA['N']) + '.pickle')).open('rb') as pkl_file:
                truth_all = pickle.load(pkl_file)

        
        return truth_all.transpose()

    elif data_name == 'observations':
        
        if set_DA['obs_type'] == 'linear':
            file_name = path / ('Y_linear_' + dt + '.pickle')
        elif set_DA['obs_type'] == 'ln':
            file_name = path / ( 'Y_ln_' + dt + '.pickle')
        elif set_DA['obs_type'] == 'abs':
            file_name = path / ( 'Y_ab_' + dt + '.pickle')
        elif set_DA['obs_type'] == 'non':
            if set_DA['J'] == 40:
                file_name = path / ( 'obs_nonlinear_' + str(set_DA['obs_sigma']) + '.pickle')
            else:
                file_name = path / ('obs_non_' + str(set_DA['J']) + '_' + str(set_DA['N'])  + '.pickle')
            # the error distribution : std: 0.5
            
        with file_name.open('rb') as pkl_file:
            ob_all = pickle.load(pkl_file)
            
        
        # if set_DA['obs_type'] == 'linear':
        #     file_name = path + '/Y_linear_' + dt + '.pickle'
        #     pkl_file = open(file_name, 'rb')
        # elif set_DA['obs_type'] == 'ln':
        #     file_name = path + '/Y_ln_' + dt + '.pickle'
        #     pkl_file = open(file_name, 'rb')
        # elif set_DA['obs_type'] == 'abs':
        #     file_name = path + '/Y_ab_' + dt + '.pickle'
        #     pkl_file = open(file_name, 'rb')
        # elif set_DA['obs_type'] == 'non':
        #     file_name = path + '/obs_nonlinear_' + \
        #         str(set_DA['obs_sigma']) + '.pickle'
        #     # the error distribution : std: 0.5
        #     pkl_file = open(file_name, 'rb')
        # ob_all = pickle.load(pkl_file)
        # pkl_file.close()
        return ob_all.transpose()

    elif data_name in ['set_DA', 'set_metrics']:
        
        with (path / data_name).open('rb') as dill_file:
            resultstatsDF_out = dill.load(dill_file)

        # with open(path + '/' + data_name, 'rb') as dill_file:
        #     resultstatsDF_out = dill.load(dill_file)

        return resultstatsDF_out
    else:
        print('Please check the data_name')

    # TODO change std ???
    # if prm._ens_new_std != None:
    #     ens = np.tile(truth_all[:, 0].reshape(prm._m, 1), prm._n) + np.random.normal(0, prm._ens_new_std,(prm._m, prm._n))
        # ens = change_ens_std(ens,prm)
    # else:
    #    pass
    # if prm._obs_new_std != None:
    #     ob_all = change_obs_std(prm)
    #
    # if prm._ens_new_std != None:
    #     ens = change_ens_std(ens,prm)


def yaml_dump(settings, set_DA,name=None, path=None):

    if path is None:
        path = Path(set_DA['data_saved_path']) / set_DA['settings_path']
    else:
        path = Path(path)

    if name is None:
        yaml_name = 'settings' + '.yaml'
    else:
        yaml_name = name + '.yaml'

    with (path / yaml_name).open('w', encoding='utf8') as outfile:
        yaml.dump(
            settings,
            outfile,
            default_flow_style=False,
            allow_unicode=True)

    # with io.open(path + '/' + yaml_name, 'w', encoding='utf8') as outfile:
    #     yaml.dump(
    #         settings,
    #         outfile,
    #         default_flow_style=False,
    #         allow_unicode=True)


def yaml_load(name, set_DA, path=None):
    
    if path is None and ('set' in name):
        path = Path(set_DA['data_saved_path']) / set_DA['settings_path']
        
    else:
        path = Path(path)
    
    with (path / ( name + '.yaml')).open('r', encoding='utf8') as outfile:
        file = yaml.load(outfile.read())

    # with io.open(path + '/' + name + '.yaml', 'r', encoding='utf8') as outfile:
    #     file = yaml.load(outfile.read())

    return file

# for LETKF


def svd0(A):
    """
    Compute the
     - full    svd if nrows > ncols
     - reduced svd otherwise.
    As in Matlab: svd(A,0),
    except that the input and output are transposed, in keeping with DAPPER convention.
    It contrasts with scipy.linalg's svd(full_matrice=False) and Matlab's svd(A,'econ'),
    both of which always compute the reduced svd.
    For reduction down to rank, see tsvd() instead.
    """
    M, N = A.shape
    if M > N:
        return sla.svd(A, full_matrices=True)
    else:
        return sla.svd(A, full_matrices=False)


def pad0(ss, N):
    out = np.zeros(N)
    out[:len(ss)] = ss
    return out

# class Nestedprint
    # import numpy as np
    # import functools
    # from IPython.lib.pretty import pretty as pretty_repr
    # import re


def filter_out(orig_list, *unwanted, INV=False):
    """
    Returns new list from orig_list with unwanted removed.
    Also supports re.search() by inputting a re.compile('patrn') among unwanted.
    """
    new = []
    for word in orig_list:
        for x in unwanted:
            try:
                # Regex compare
                rm = x.search(word)
            except AttributeError:
                # String compare
                rm = x == word
            if (not INV) == bool(rm):
                break
        else:
            new.append(word)
    return new


# Local np.set_printoptions. stackoverflow.com/a/2891805/38281
@contextlib.contextmanager
@functools.wraps(np.set_printoptions)
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def repr_type_and_name(thing):
    """Print thing's type [and name]"""
    s = "<" + type(thing).__name__ + '>'
    if hasattr(thing, 'name'):
        s += ': ' + thing.name
    return s


class NestedPrint:

    """
    Multi-Line, Recursive repr (print) functionality.
    Set class variables to change look:
     - 'indent': indentation per level
     - 'ch': character to use for "spine" (e.g. '|' or ' ')
     - 'ordr_by_linenum': 0: alphabetically, 1: linenumbr, -1: reverse
    """
    indent = 3
    ch = '.'
    ordr_by_linenum = 0

    # numpy print options
    threshold = 10
    precision = None

    # Recursion monitoring.
    _stack = []  # Reference using NestedPrint._stack, ...
    # not self._stack or self.__class__, which reference sub-class "instance".

    # Reference using self.excluded, to access sub-class "instance".
    excluded = []  # Don't include in printing
    excluded.append(re.compile('^_'))  # "Private"
    excluded.append('name')  # Treated separately

    included = []  # Only print these (also determines ordering).
    ordering = []  # Determine ordering (with precedence over included).
    aliases = {}  # Rename attributes (vars)

    def __repr__(self):
        with printoptions(threshold=self.threshold, precision=self.precision):
            # new line chars
            NL = '\n' + self.ch + ' ' * (self.indent - 1)

            # Infinite recursion prevention
            is_top_level = False
            if NestedPrint._stack == []:
                is_top_level = True
            if self in NestedPrint._stack:
                return "**Recursion**"
            NestedPrint._stack += [self]

            # Use included or filter-out excluded
            keys = self.included or filter_out(vars(self), *self.excluded)

            # Process attribute repr's
            txts = {}
            for key in keys:
                t = pretty_repr(getattr(self, key))  # sub-repr
                if '\n' in t:
                    # Activate multi-line printing
                    t = t.replace(
                        '\n', NL + ' ' * self.indent)      # other lines
                    t = NL + ' ' * self.indent + t                  # first line
                t = NL + self.aliases.get(key, key) + ': ' + t  # key-name
                txts[key] = t

            def sortr(x):
                if x in self.ordering:
                    key = -1000 + self.ordering.index(x)
                else:
                    if self.ordr_by_linenum:
                        key = self.ordr_by_linenum * txts[x].count('\n')
                    else:
                        key = x.lower()
                        # Convert str to int (assuming ASCII) for comparison
                        # with above cases
                        key = sum(ord(x) * 128**i for i,
                                  x in enumerate(x[::-1]))
                return key

            # Assemble string
            s = repr_type_and_name(self)
            for key in sorted(txts, key=sortr):
                s += txts[key]

            # Empty _stack when top-level printing finished
            if is_top_level:
                NestedPrint._stack = []

            return s


class container(NestedPrint):
        # to transfer a dict into a class
    def __init__(self, obs):
        for key, value in obs.items():
            setattr(self, key, value)

# class FAU_series


class AssimFailedError(RuntimeError):
    pass


def is_int(a):
    return np.issubdtype(type(a), np.integer)


def weight_degeneracy(w, prec=1e-10):
    return (1 - w.max()) < prec


def unbias_var(w=None, N_eff=None, avoid_pathological=False):
    """
    Compute unbias-ing factor for variance estimation.
    wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
    """
    if N_eff is None:
        N_eff = 1 / (w@w)
    if avoid_pathological and weight_degeneracy(w):
        ub = 1  # Don't do in case of weights collapse
    else:
        ub = 1 / (1 - 1 / N_eff)  # =N/(N-1) if w==ones(N)/N.
    return ub


def raise_AFE(msg, time_index=None):
    if time_index is not None:
        msg += "\n(k,kObs,fau) = " + str(time_index) + ". "
    raise AssimFailedError(msg)


class FAU_series(NestedPrint):
    """
    Container for time series of a statistic from filtering.
    Data is indexed with key (k,kObs,f_a_u) or simply k.
    The accessing then categorizes the result as
     - forecast   (.f, len: KObs+1)
     - analysis   (.a, len: KObs+1)
     - smoothed   (.s, len: KObs+1)
     - universial (.u, len: K+1)
       - also contains time instances where there are no obs.
         These intermediates are nice for plotting.
       - may also be hijacked to store "smoothed" values.
    Data may also be accessed through raw attributes [.a, .f, .s, .u].
    NB: if time series is only from analysis instances (len KObs+1),
        then you should use a simple np.array instead.
    """

    # Printing options (cf. NestedPrint)
    # included = NestedPrint.included + ['f', 'a', 's', 'store_u']

    included = NestedPrint.included + ['f', 'a', 'store_u']
    aliases = {
        'f': 'Forecast (.f)',
        'a': 'Analysis (.a)',
        's': 'Smoothed (.s)',
        'u': 'All      (.u)'}
    aliases = {**NestedPrint.aliases, **aliases}

    def __init__(self, set_DA, M, store_u=True, **kwargs):
        """
        Constructor.
         - chrono  : a Chronology object.
         - M       : len (or shape) of items in series.
         - store_u : if False: only the current value is stored.
         - kwargs  : passed on to ndarrays.
        """

        self.store_u = store_u

        # self.chrono = chrono

        prm = container(set_DA)
        da_time_len = prm.da_time_len
        update_times = prm.update_times

        # Convert int-len to shape-tuple
        self.M = M  # store first
        if is_int(M):
            if M == 1:
                M = ()
            else:
                M = (M,)

        self.a = np.full((update_times,) + M, nan, **kwargs)
        self.f = np.full((update_times,) + M, nan, **kwargs)
        self.s = np.full((update_times,) + M, nan, **kwargs)
        if self.store_u:
            self.u = np.full((da_time_len,) + M, nan, **kwargs)
        else:
            self.tmp = np.full(M, nan, **kwargs)
            self.k_tmp = None

    def validate_key(self, key):
        try:
            # Assume key = (k,kObs,fau)
            if not isinstance(key, tuple):
                raise ValueError
            k, kObs, fau = key  # Will also raise ValueError
            if not isinstance(fau, str):
                raise ValueError
            if not all([letter in 'fasu' for letter in fau]):
                raise ValueError
            #
            if kObs is None:
                for ltr in 'afs':
                    if ltr in fau:
                        raise KeyError(
                            "Accessing ." + ltr + " series, but kObs is None.")
            # NB: The following check has been disabled, because
            # it is actually very time consuming when kkObs is long (e.g. 10**4):
            # elif k != self.chrono.kkObs[kObs]: raise KeyError("kObs
            # indicated, but k!=kkObs[kObs]")
        except ValueError:
            # Assume key = k
            assert not hasattr(
                key, '__getitem__'), "Key must be 1-dimensional."
            key = (key, None, 'u')
        return key

    def split_dims(self, k):
        "Split (k,kObs,fau) into k, (kObs,fau)"
        if isinstance(k, tuple):
            k1 = k[1:]
            k0 = k[0]
        elif is_int(k):
            k1 = ...
            k0 = k
        else:
            raise KeyError
        return k0, k1

    def __setitem__(self, key, item):
        k, kObs, fau = self.validate_key(key)
        if 'f' in fau:
            self.f[kObs] = item
        if 'a' in fau:
            self.a[kObs] = item
        if 's' in fau:
            self.s[kObs] = item
        if 'u' in fau:
            if self.store_u:
                self.u[k] = item
            else:
                k0, k1 = self.split_dims(k)
                self.k_tmp = k0
                self.tmp[k1] = item

    def __getitem__(self, key):
        k, kObs, fau = self.validate_key(key)

        if len(fau) > 1:
            # Check consistency. NB: Somewhat time-consuming.
            for sub in fau[1:]:
                i1 = self[k, kObs, sub]
                i2 = self[k, kObs, fau[0]]
                if np.any(i1 != i2):
                    if not (np.all(np.isnan(i1)) and np.all(np.isnan(i2))):
                        raise RuntimeError(
                            "Requested item corresponding to multiple arrays ('%s'), " %
                            fau + "But the items are not equal.")

        if 'f' in fau:
            return self.f[kObs]
        elif 'a' in fau:
            return self.a[kObs]
        elif 's' in fau:
            return self.s[kObs]
        else:
            if self.store_u:
                return self.u[k]
            else:
                k0, k1 = self.split_dims(k)
                if self.k_tmp != k0:
                    msg = "Only item [" + str(self.k_tmp) + "] is available from " + \
                          "the universal (.u) series. One possible source of error " + \
                          "is that the data has not been computed for entry k=" + str(k0) + ". " + \
                          "Another possibility is that it has been cleared; " + \
                          "if so, a fix might be to set store_u=True, " + \
                          "or to use analysis (.a), forecast (.f), or smoothed (.s) arrays instead."
                    raise KeyError(msg)
                return self.tmp[k1]

    # def average(self):
    #     """
    #     Avarage series,
    #     but only if it's univariate (scalar).
    #     """
    #     if self.M > 1:
    #         raise NotImplementedError
    #     avrg = {}
    #     t = self.chrono
    #     for sub in 'afsu':
    #         if sub == 'u':
    #             inds = t.kk[t.mask_BI]
    #         else:
    #             inds = t.maskObs_BI
    #         if hasattr(self, sub):
    #             series = getattr(self, sub)[inds]
    #             avrg[sub] = series_mean_with_conf(series)
    #     return avrg

    def __repr__(self):
        if self.store_u:
            # Create instance version of 'included'
            self.included = self.included + ['u']
        return super().__repr__()

# class FAU_series


class evaluation():

    def __init__(self, set_DA, set_metrics, read_or_not=None):

        self.read_or_not = read_or_not

        # metrics_list = ['mu', 'var', 'mad', 'err', 'logp_m', 'skew', 'kurt', 'rmv', 'rmse', 'rh', 'svals', 'umisf']

        self.metrics_list = []

        for key, value in set_metrics.items():
            setattr(self, key, value)
            self.metrics_list.append(key)

        for key, value in set_DA.items():
            setattr(self, key, value)

    def ens_valuate(self, t, truth=None, w=None, type=None, E=None):

        self.truth = truth

        if (self.read_or_not is None) and (
                type in ['analysis', 'forecast', 'all']) and (E is not None):

            self.write_to_attribute(t, E, w, type)

        elif (self.read_or_not == 'read') and (type in ['analysis', 'forecast', 'all']):

            if type == 'analysis':
                pass  # read ens
            elif type == 'forecast':
                pass  # read ens
            elif type == 'all':
                pass  # read ens

            # TOdo self.write_to_attribute(t, E, w, type)

    def write_to_attribute(self, tt, E, w, type):

        if type == 'analysis':
            atr = 'a'
            t = np.where(self.update_time_inds == tt)
        elif type == 'forecast':
            atr = 'f'
            t = np.where(self.update_time_inds == tt)
        elif type == 'all':
            atr = 'u'
            t = np.where(self.da_t_inds == tt)

        self.ens_cal(E, t, w)

        # print(t)

        for metrics in self.metrics_list:
            get_attribute = getattr(getattr(self, metrics), atr)
            tmp = getattr(self, metrics + '_')

            # print(metrics)
            # print(get_attribute.shape)
            # print(tmp)

            if tmp.ndim == 0:
                get_attribute[t] = tmp
                # print(get_attribute[t])
            else:
                get_attribute[t, :] = tmp
                # print(get_attribute[t, :])

    def ens_cal(self, E, t, w=None):

        # Validate weights
        if w is None:
            try:
                delattr(self, 'w')
            except AttributeError:
                pass
            finally:
                w = 1 / self.N
        if np.isscalar(w):
            assert w != 0
            w = w * np.ones(self.N)
        if hasattr(self, 'w'):
            w = self.w  # self.w = w

        if abs(w.sum() - 1) > 1e-5:
            raise_AFE("Weights did not sum to one.", t)
        if not np.all(np.isfinite(E)):
            raise_AFE("Ensemble not finite.", t)
        if not np.all(np.isreal(E)):
            raise_AFE("Ensemble not Real.", t)

        if hasattr(self, 'mu'):
            self.mu_ = w @ E

        A = E - self.mu_

        if hasattr(self, 'mad'):
            self.mad_ = w @ abs(A)  # Mean abs deviations

        if hasattr(self, 'var'):
            # While A**2 is approx as fast as A*A,
            # A**3 is 10x slower than A**2 (or A**2.0).
            # => Use A2 = A**2, A3 = A*A2, A4=A*A3.
            # But, to save memory, only use A_pow.
            A_pow = A ** 2

            self.var_ = w @ A_pow

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            ub = unbias_var(w, avoid_pathological=True)
            self.var_ *= ub

        if hasattr(self, 'var') and hasattr(self, 'skew'):
            # For simplicity, use naive (biased) formulae, derived
            # from "empirical measure". See doc/unbiased_skew_kurt.jpg.
            # Normalize by var. Compute "excess" kurt, which is 0 for
            # Gaussians.
            A_pow *= A
            self.skew_ = np.nanmean(w @ A_pow / self.var_ ** (3 / 2))

        if hasattr(self, 'skew') and hasattr(self, 'kurt'):
            A_pow *= A  # idem.
            self.kurt_ = np.nanmean(w @ A_pow / self.var_ ** 2 - 3)

        if hasattr(self, 'err'):
            self.err_ = self.mu_ - self.truth

        if hasattr(self, 'rmv') and hasattr(self, 'var'):
            self.rmv_ = np.sqrt(np.mean(self.var_))

        if hasattr(self, 'rmse') and hasattr(self, 'err'):
            self.rmse_ = np.sqrt(np.mean(self.err_ ** 2))

        if hasattr(
                self,
                'logp_m') and hasattr(
                self,
                'var') and hasattr(
                self,
                'err'):
            Nx = len(self.err_)
            ldet = np.log(self.var_).sum()
            nmisf = self.var_ ** (-1 / 2) * self.err_
            logp_m = (nmisf ** 2).sum() + ldet
            self.logp_m_ = logp_m / Nx

        if hasattr(self, 'svals') and hasattr(self, 'umisf'):
            if self.N <= self.J:
                _, s, UT = svd((sqrt(w) * A.T).T, full_matrices=False)
                s *= sqrt(ub)  # Makes s^2 unbiased
                self.svals_ = s
                self.umisf_ = UT @ self.err_
            else:
                P = (A.T * w) @ A
                s2, U = eigh(P)
                s2 *= ub
                self.svals_ = sqrt(s2.clip(0))[::-1]
                self.umisf_ = U.T[::-1] @ self.err_

        if hasattr(self, 'rh'):

            # For each state dim [i], compute rank of truth (x) among the
            # ensemble (E)
            Ex_sorted = np.sort(np.vstack((E, self.truth)),
                                axis=0, kind='heapsort')
            self.rh_ = np.array([np.where(Ex_sorted[:, i] == self.truth[i])[
                                0][0] for i in range(self.J)])

        if hasattr(self, 'normaltest'):
            E_shape1, E_shape2 = E.shape
            if E_shape1 <10:
                self.normaltest_ = nan
            else:
                self.normaltest_ = ss.normaltest(E, axis=0)[1]

        if hasattr(self,'x1'):
            self.x1_ = E[:,0]

        if hasattr(self,'x2'):
            self.x2_ = E[:,1]

# resampling methods for PF


def resampleSystematic(w, N):

    M = len(w)
    w = w / np.sum(w)
    Q = np.cumsum(w)
    indx = np.zeros(N)
    T = np.linspace(0, 1 - 1 / (N), (N)) + np.random.rand(1) / N  # 0.007

    i = 0
    j = 0
    while (i <= N - 1 and j <= M - 1):
        while Q[j] < T[i]:
            j = j + 1
        indx[i] = j
        i = i + 1

    return indx


def random_residual(w, N=None):
    # np.random.seed(1000)
    if N is None:
        N = len(w)
    M = len(w)
    w = w / sum(w)
    indx = []

    Ns = np.floor(np.multiply(N, w))
    R = np.sum(Ns)
    j = 0
    while j < M:
        cnt = 1
        while cnt <= Ns[j]:
            indx.append(j)
            cnt = cnt + 1
        j = j + 1

    N_rdn = N - R
    Ws = (N * w - Ns) / N_rdn
    Q = np.cumsum(Ws)
    i = len(indx)
    while i < N:
        sampl = np.random.rand()  # 0.55
        # print('sampl',sampl)
        j = 0
        while Q[j] < sampl:
            j = j + 1
        indx.append(j)
        i = i + 1
    indx = np.array(indx)

    # print(i)

    return indx


# for adaptive inflation method (Anderson 2007)
def solve_quadratic(a, b, c):
    scaling = max(abs(a), abs(b), abs(c))
    a_s = a / scaling
    bs = b / scaling
    cs = c / scaling

    disc = math.sqrt(bs ** 2 - 4.0 * a_s *cs)

    if ( bs > 0.0 ):
        r1 = (-bs - disc) / (2 * a_s)
    else:
        r1 = (-bs + disc) / (2 * a_s)

    r2 = (cs / a_s) / r1

    return r1,r2

def linear_bayes(dist_2, sigma_p_2, sigma_o_2, lambda_mean, lambda_sd_2, gamma):

    theta_bar_2 = (1.0 + gamma * (math.sqrt(lambda_mean) - 1.0)) ** 2 * sigma_p_2 + sigma_o_2
    theta_bar = math.sqrt(theta_bar_2)
    u_bar = 1.0 / (math.sqrt(2.0 * math.pi) * theta_bar)

    like_exp_bar = dist_2 / (-2.0 * theta_bar_2)
    v_bar = math.exp(like_exp_bar)
    like_bar = u_bar * v_bar

    if (like_bar <= 0.0):

        new_cov_inflate = lambda_mean
    else:
        dtheta_dlambda = 0.5 * sigma_p_2 * gamma * (1.0 - gamma + gamma * math.sqrt(lambda_mean)) / (theta_bar * math.sqrt(lambda_mean))
        like_prime = (u_bar * v_bar * dtheta_dlambda / theta_bar) * (dist_2 / theta_bar_2 - 1.0)

        if (like_prime == 0.0):
            new_cov_inflate = lambda_mean
        else:
            a = 1.0
            b = like_bar / like_prime - 2.0 * lambda_mean
            c = lambda_mean ** 2 - lambda_sd_2 - like_bar * lambda_mean / like_prime

            plus_root,minus_root = solve_quadratic(a, b, c)

            if (abs(minus_root - lambda_mean) < abs(plus_root - lambda_mean)):
                new_cov_inflate = minus_root
            else:
                new_cov_inflate = plus_root

    return new_cov_inflate

def bayes_cov_inflate(x_p, sigma_p_2, y_o, sigma_o_2, lambda_mean, lambda_sd, gamma, sd_lower_bound_in = 0.6):

    # ! Uses algorithms in references on DART web site to update the distribution of inflation.
    #
    # real(r8), intent(in)  :: x_p, sigma_p_2, y_o, sigma_o_2, lambda_mean, lambda_sd, gamma
    # real(r8), intent(in)  :: sd_lower_bound_in
    # real(r8), intent(out) :: new_cov_inflate, new_cov_inflate_sd
    #
    # integer  :: i, mlambda_index(1)
    #
    # real(r8) :: new_1_sd, new_max, ratio, lambda_sd_2
    # real(r8) :: dist_2, b, c, d, Q, R, disc, alpha, beta, cube_root_alpha, cube_root_beta, x
    # real(r8) :: rrr, cube_root_rrr, angle, mx(3), sep(3), mlambda(3)

    # ! If gamma is 0, nothing happens
    if (gamma <= 0.0):
        new_cov_inflate = lambda_mean
        new_cov_inflate_sd = lambda_sd
        # return new_cov_inflate,new_cov_inflate_sd

    else:
        mx = np.zeros(3)
        sep = np.zeros(3)
        mlambda = np.zeros(3)

        # ! Computation saver
        lambda_sd_2 = lambda_sd ** 2
        dist_2 = (x_p - y_o) ** 2

        if (gamma > 1.01):
            b = -1.0 * (sigma_o_2 + sigma_p_2 * lambda_mean)
            c = lambda_sd_2 * sigma_p_2 ** 2 / 2.0
            d = -1.0 * (lambda_sd_2 * sigma_p_2 ** 2 * dist_2) / 2.0

            Q = c - b ** 2 / 3
            R = d + (2 * b ** 3) / 27 - (b * c) / 3

            disc = R ** 2 / 4 + Q ** 3 / 27

            if (disc < 0.0):
                rrr = math.sqrt(-1.0 * Q ** 3 / 27)
                cube_root_rrr = rrr ** (1.0 / 3.0)
                angle = math.acos(-0.5 * R / rrr)

                for i in range(0,3):
                    mx[i] = 2.0 * cube_root_rrr * math.cos((angle + i * 2.0 * math.pi) / 3.0) - b / 3.0

                    mlambda[i] = (mx[i] - sigma_o_2) / sigma_p_2
                    sep[i]= abs(mlambda[i] - lambda_mean)

                mlambda_index = np.argmin(sep)
                new_cov_inflate = mlambda[mlambda_index]
            else:
                alpha = -R / 2 +  math.sqrt(disc)
                beta = R / 2 + math.sqrt(disc)

                cube_root_alpha = abs(alpha) ** (1.0 / 3.0) * abs(alpha) / alpha
                cube_root_beta = abs(beta) ** (1.0 / 3.0) * abs(beta) / beta

                x = cube_root_alpha - cube_root_beta - b / 3.0

                new_cov_inflate = (x - sigma_o_2) / sigma_p_2

        else:
            new_cov_inflate = linear_bayes(dist_2, sigma_p_2, sigma_o_2, lambda_mean, lambda_sd_2, gamma)

        if (lambda_sd <= sd_lower_bound_in):
            new_cov_inflate_sd = lambda_sd
        else:
            new_max = compute_new_density(dist_2, sigma_p_2, sigma_o_2, lambda_mean, lambda_sd, gamma, new_cov_inflate)
            new_1_sd = compute_new_density(dist_2, sigma_p_2, sigma_o_2, lambda_mean, lambda_sd, gamma, new_cov_inflate + lambda_sd)

            if ( abs(new_max) <= 0.0  or abs(new_1_sd) <= 0.0 ):
                new_cov_inflate_sd = lambda_sd
                # return new_cov_inflate, new_cov_inflate_sd
            else:
                ratio = new_1_sd / new_max
                if (ratio > 0.99):
                    new_cov_inflate_sd = lambda_sd
                    # return new_cov_inflate, new_cov_inflate_sd
                else:
                    new_cov_inflate_sd = math.sqrt(-1.0 * lambda_sd_2 / (2.0 * math.log(ratio)))

                    if (new_cov_inflate_sd > lambda_sd):
                        new_cov_inflate_sd = lambda_sd

    return new_cov_inflate, new_cov_inflate_sd

def compute_new_density(dist_2, sigma_p_2, sigma_o_2, lambda_mean, lambda_sd, gamma, lambuda):

    # Used to update density by taking approximate gaussian product
    # real(r8)             :: compute_new_density
    # real(r8), intent(in) :: dist_2, sigma_p_2, sigma_o_2, lambda_mean, lambda_sd, gamma, lambda
    # real(r8) :: theta_2, theta
    # real(r8) :: exponent_prior, exponent_likelihood

    # Compute probability of this lambda being correct
    exponent_prior = (lambuda - lambda_mean)**2 / (-2.0 * lambda_sd**2)

    # Compute probability that observation would have been observed given this lambda
    # print(lambuda)
    theta_2 = (1.0 + gamma * (math.sqrt(lambuda) - 1.0))**2 * sigma_p_2 + sigma_o_2

    theta = math.sqrt(theta_2)

    exponent_likelihood = dist_2 / ( -2.0 * theta_2)
    # Compute the updated probability density for lambda
    # Have 1 / sqrt(2 PI) twice, so product is 1 / (2 PI)
    compute_new_density = np.exp(exponent_likelihood + exponent_prior) / (2.0 * math.pi * lambda_sd * theta)
    #end function compute_new_density

    return compute_new_density

def inflate_ens(ens,inflate,name = 'deterministic'):

    mean = ens.mean(axis = 0)
    var = ens.var(axis = 0)

    ens_size, model_size = ens.shape

    if name == 'deterministic':

        sd_inflate = math.sqrt(inflate)
        ens = ens * sd_inflate + np.tile(mean,(ens_size,1)) * (1.0 - sd_inflate)
    else:
        # stochastic algorithm

        if (inflate > 1.0):
            rand_sd = np.sqrt(inflate * var - var)
            ens = np.random.normal(0,rand_sd,ens.shape)

        ens = ens - np.tile( (ens.sum(axis = 0) / ens_size - mean),(ens_size,1))

    return ens

def ob_space_infl(prm, ob, HE_mean, HE_var, gamma=1):

    # x_p = HE_mean[0]
    # y_o = ob[0]
    # sigma_p_2 = HE_var[0]
    # sigma_o_2 = 1
    # bayes_cov_inflate(x_p, sigma_p_2, y_o, sigma_o_2, lambda_mean, lambda_sd, gamma, sd_lower_bound_in=0.6)
    # bayes_cov_inflate(x_p, sigma_p_2, y_o, sigma_o_2, 1.02, 0.1, 1, sd_lower_bound_in=0.5)
    inf_lower_bound = 1.0
    inf_upper_bound = 2
    sd_lower_bound = 0.6

    prm = container(prm)

    for i in range(len(ob)):
        # print(i)
        x_p = HE_mean[i]
        y_o = ob[i]
        sigma_p_2 = HE_var[i]
        sigma_o_2 = prm.obs_sigma

        if i == 0:
            lambda_mean = 1.02
            lambda_sd = 1

        lambda_mean, lambda_sd = bayes_cov_inflate(x_p, sigma_p_2, y_o, sigma_o_2, lambda_mean, lambda_sd,
                                                   gamma, sd_lower_bound_in=0.6)

        if (lambda_mean < inf_lower_bound):
            lambda_mean = inf_lower_bound
        if (lambda_mean > inf_upper_bound):
            lambda_mean = inf_upper_bound
        if (lambda_sd < sd_lower_bound):
            lambda_sd = sd_lower_bound

    return lambda_mean, lambda_sd



def infl_ens(set_DA,ens, ob, HE):

    ens_a, ens_b = ens.shape
    if ens_a == set_DA['N']:
        pass
    else:
        ens = ens.T

    if set_DA['adaptive_infl']:
        HE_a ,HE_b = HE.shape
        if HE_a == set_DA['N']:
            pass
        else:
            HE = HE.T
        HE_mean = HE.mean(axis = 0)
        HE_var = HE.var(axis = 0)
        lambda_mean, lambda_sd = ob_space_infl(set_DA, ob, HE_mean, HE_var, gamma=1)
    else:
        lambda_mean = set_DA['infl']

    # print(lambda_mean)

    if lambda_mean>1.0:
        ens_new = inflate_ens(ens, lambda_mean, name='deterministic')
    else:
        ens_new = ens

    return ens_new.T



# class GT
class GT:
    _name = 'Python version of Gamma Test'

    def __init__(self, x, y, p=10):
        self._x = x
        self._y = y
        self._p = p
        self.length = len(self._x)

        self._pts = list(zip(self._x.ravel(), np.zeros(len(self._x)).ravel()))

    def initial(self):
        self._x = np.random.normal(0, 1, 100)
        self._y = np.random.normal(0, 1, 100)
        self._pts = list(zip(self._x.ravel(), np.zeros(len(self._x)).ravel()))


    def knn_find(self):
        self._Dis = np.zeros((10, len(self._x)))
        self._idx = np.zeros((10, len(self._x)), dtype=int)

        for k in range(len(self._x)):

            tmp_in_knn = self._x[k]
            self._d = (self._x - tmp_in_knn) * (self._x - tmp_in_knn)
            idx_t = np.argsort(self._d)
            self._idx[:, k] = idx_t[1:11]
            # Dis_t = np.sort(self._d)
            self._Dis[:, k] = self._d[self._idx[:, k]]


        return self._idx.transpose(), self._Dis.transpose()


    def knn_find_simple(self):

        x_v = self._x*np.ones((self.length,self.length))
        self._d = np.square(x_v - x_v.T)

        # x_v = np.tile(self._x[:, None], self.length)
        # x_h = np.tile(self._x[None, :], (self.length,1))
        # self._d = np.square(x_v - x_h)
        d_indx = np.argsort(self._d,axis=1)
        self._idx = d_indx[:,1:11]
        d = np.sort(self._d, axis=1)
        self._Dis = d[:, 1:11]

        return self._idx, self._Dis

    def knn_find_simple_slice(self):

        self._Dis = np.zeros((self.length,10))
        x_v = np.tile(self._x[:, None], self.length)
        x_h = np.tile(self._x[None, :], (self.length,1))

        self._d = np.square(x_v - x_h)
        d_indx = np.argsort(self._d,axis=1)
        self._idx = d_indx[:,1:11]

        for i in range(self.length):
            self._Dis[i,:] = self._d[i,self._idx[i,:]]

        return self._idx, self._Dis

    def Gamma_knn(self):

        nni, nnd = self.knn_find()
        if sum(sum(nni)) ==  0:
            print('Gamma test failed---------------------------------------------')

        M, p = nni.shape
        gamma = np.zeros((M, p))
        self._Delta = np.zeros((M, p))
        for i in range(M):
            tmp_in_GT = self._y[i]
            for k in range(p):
                gamma[i, k] = 0.5 * np.square( tmp_in_GT- self._y[nni[i, k]])



        self._Gamma = np.mean(gamma, 0)
        self._Delta = np.mean(nnd, 0)

        rline = np.polyfit(self._Delta, self._Gamma, 1)
        self.GT = rline[1]


        return self.GT

    def Gamma_knn_simple(self):


        nni, nnd = self.knn_find_simple()
        if sum(sum(nni)) ==  0:
            print('Gamma test failed---------------------------------------------')

        M, p = nni.shape
        gamma = np.zeros((M, p))

        y_v = (self._y[:, None] * np.ones((self.length, self._p)))

        #y_v = np.tile(self._y[:, None], self._p)
        gamma = 0.5 * np.square(y_v - self._y[nni])


        self._Gamma = np.mean(gamma, 0)
        self._Delta = np.mean(nnd, 0)

        rline = np.polyfit(self._Delta, self._Gamma, 1)
        self.GT = rline[1]


        return self.GT

    def Gamma_knn_simple_slice(self):

        nni, nnd = self.knn_find_simple_slice()
        if sum(sum(nni)) ==  0:
            print('Gamma test failed---------------------------------------------')

        M, p = nni.shape
        gamma = np.zeros((M, p))

        y_v = np.tile(self._y[:, None], self._p)
        gamma = 0.5 * np.square(y_v - self._y[nni])


        self._Gamma = np.mean(gamma, 0)
        self._Delta = np.mean(nnd, 0)

        rline = np.polyfit(self._Delta, self._Gamma, 1)
        self.GT = rline[1]


        return self.GT


# others
# to send an email
def send_email(user, pwd, recipient, subject, body):

    # send_email('niu09024@gmail.com', 'yxukxlygrpcqaajp', 'niu09024@163.com', 'trytry', 'everything is done')

    import smtplib

    FROM = user
    TO = recipient if isinstance(recipient, list) else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(user, pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print('successfully sent the mail')
    except:
        print("failed to send mail")


# observations cov
class RV(NestedPrint):
    "Class to represent random variables."

    # Used by NestedPrint
    ordr_by_linenum = +1

    def __init__(self, M, **kwargs):
        """
         - M    <int>     : ndim
         - is0  <bool>    : if True, the random variable is identically 0
         - func <func(N)> : use this sampling function. Example:
                            RV(M=4,func=lambda N: rand((N,4))
         - file <str>     : draw from file. Example:
                            RV(M=4,file=data_dir+'/tmp.npz')
        The following kwords (versions) are available,
        but should not be used for anything serious (use instead subclasses, like GaussRV).
         - icdf <func(x)> : marginal/independent  "inverse transform" sampling. Example:
                            RV(M=4,icdf = scipy.stats.norm.ppf)
         - cdf <func(x)>  : as icdf, but with approximate icdf, from interpolation. Example:
                            RV(M=4,cdf = scipy.stats.norm.cdf)
         - pdf  <func(x)> : "acceptance-rejection" sampling
                            Not implemented.
        """
        self.M = M
        for key, value in kwargs.items():
            setattr(self, key, value)

    def sample(self, N):
        if getattr(self, 'is0', False):
            # Identically 0
            E = zeros((N, self.M))
        elif hasattr(self, 'func'):
            # Provided by function
            E = self.func(N)
        elif hasattr(self, 'file'):
            # Provided by numpy file with sample
            data = np.load(self.file)
            sample = data['sample']
            N0 = len(sample)
            if 'w' in data:
                w = data['w']
            else:
                w = ones(N0) / N0
            idx = np.random.choice(N0, N, replace=True, p=w)
            E = sample[idx]
        elif hasattr(self, 'icdf'):
            # Independent "inverse transform" sampling
            icdf = np.vectorize(self.icdf)
            uu = rand((N, self.M))
            E = icdf(uu)
        elif hasattr(self, 'cdf'):
            # Like above, but with inv-cdf approximate, from interpolation
            if not hasattr(self, 'icdf_interp'):
                # Define inverse-cdf
                from scipy.interpolate import interp1d
                from scipy.optimize import fsolve
                cdf = self.cdf
                Left, = fsolve(lambda x: cdf(x) - 1e-9, 0.1)
                Right, = fsolve(lambda x: cdf(x) - (1 - 1e-9), 0.1)
                xx = linspace(Left, Right, 1001)
                uu = np.vectorize(cdf)(xx)
                icdf = interp1d(uu, xx)
                self.icdf_interp = np.vectorize(icdf)
            uu = rand((N, self.M))
            E = self.icdf_interp(uu)
        elif hasattr(self, 'pdf'):
            # "acceptance-rejection" sampling
            raise NotImplementedError
        else:
            raise KeyError
        assert self.M == E.shape[1]
        return E

def exactly_1d(a):
  a = np.atleast_1d(a)
  assert a.ndim==1
  return a
def exactly_2d(a):
  a = np.atleast_2d(a)
  assert a.ndim==2
  return a
class RV_with_mean_and_cov(RV):
    """Generic multivariate random variable characterized by two parameters: mean and covariance.
    This class must be subclassed to provide sample(),
    i.e. its main purpose is provide a common convenience constructor.
    """

    def __init__(self, mu=0, C=0, M=None):
        """Init allowing for shortcut notation."""

        if isinstance(mu, CovMat):
            raise TypeError("Got a covariance paramter as mu. " +
                            "Use kword syntax (C=...) ?")

        # Set mu
        mu = exactly_1d(mu)
        if len(mu) > 1:
            if M is None:
                M = len(mu)
            else:
                assert len(mu) == M
        else:
            if M is not None:
                mu = ones(M) * mu

        # Set C
        if isinstance(C, CovMat):
            if M is None:
                M = C.M
        else:
            if C is 0:
                pass  # Assign as pure 0!
            else:
                if np.isscalar(C):
                    M = len(mu)
                    C = CovMat(C * ones(M), 'diag')
                else:
                    C = CovMat(C)
                    if M is None:
                        M = C.M

        # Validation
        if len(mu) not in (1, M):
            raise TypeError("Inconsistent shapes of (M,mu,C)")
        if M is None:
            raise TypeError("Could not deduce the value of M")
        try:
            if M != C.M:
                raise TypeError("Inconsistent shapes of (M,mu,C)")
        except AttributeError:
            pass

        # Assign
        self.M = M
        self.mu = mu
        self.C = C

    def sample(self, N):
        """Sample N realizations. Returns N-by-M (ndim) sample matrix.
        Example::
          plt.scatter(*(UniRV(C=randcov(2)).sample(10**4).T))
        """
        if self.C is 0:
            D = zeros((N, self.M))
        else:
            D = self._sample(N)
        return self.mu + D

    def _sample(self, N):
        raise NotImplementedError("Must be implemented in subclass")

def truncate_rank(s,threshold,avoid_pathological):
  "Find r such that s[:r] contains the threshold proportion of s."
  assert isinstance(threshold,float)
  if threshold == 1.0:
    r = len(s)
  elif threshold < 1.0:
    r = np.sum(np.cumsum(s)/np.sum(s) < threshold)
    r += 1 # Hence the strict inequality above
    if avoid_pathological:
      # If not avoid_pathological, then the last 4 diag. entries of
      # reconst( *tsvd(eye(400),0.99) )
      # will be zero. This is probably not intended.
      r += np.sum(np.isclose(s[r-1], s[r:]))
  else:
    raise ValueError
  return r

class lazy_property(object):
    '''
    # From stackoverflow.com/q/3012421
    Lazy evaluation of property.
    Should represent non-mutable data,
    as it replaces itself.
    '''
    def __init__(self,fget):
      self.fget = fget
      self.func_name = fget.__name__

    def __get__(self,obj,cls):
      value = self.fget(obj)
      setattr(obj,self.func_name,value)
      return value

class CovMat():
    """Covariance matrix class.
    Main tasks:
      - Unifying the covariance representations:
        full, diagonal, reduced-rank sqrt.
      - Convenience constructor and printing.
      - Convenience transformations with memoization.
        E.g. replaces:
        >if not hasattr(noise.C,'sym_sqrt'):
        >  S = funm_psd(noise.C, sqrt)
        >  noise.C.sym_sqrt = S
        This (hiding it internally) becomes particularly useful
        if the covariance matrix changes with time (but repeat).
    """

    ##################################
    # Init
    ##################################
    def __init__(self, data, kind='full_or_diag', trunc=1.0):
        """The covariance (say P) can be input (specified in the following ways):
        kind    : data
        ----------------------
        'full'  : full M-by-M array (P)
        'diag'  : diagonal of P (assumed diagonal)
        'E'     : ensemble (N-by-M) with sample cov P
        'A'     : as 'E', but pre-centred by mean(E,axis=0)
        'Right' : any R such that P = R.T@R (e.g. weighted form of 'A')
        'Left'  : any L such that P = L@L.T
        """

        # Cascade if's down to 'Right'
        if kind == 'E':
            mu = mean(data, 0)
            data = data - mu
            kind = 'A'
        if kind == 'A':
            N = len(data)
            data = data / sqrt(N - 1)
            kind = 'Right'
        if kind == 'Left':
            data = data.T
            kind = 'Right'
        if kind == 'Right':
            # If a cholesky factor has been input, we will not
            # automatically go for the EVD, seeing as e.g. the
            # diagonal can be computed without it.
            R = exactly_2d(data)
            self._R = R
            self._m = R.shape[1]
        else:
            if kind == 'full_or_diag':
                data = np.atleast_1d(data)
                if data.ndim == 1 and len(data) > 1:
                    kind = 'diag'
                else:
                    kind = 'full'
            if kind == 'full':
                # If full has been imput, then we have memory for an EVD,
                # which will probably be put to use in the DA.
                C = exactly_2d(data)
                self._C = C
                M = len(C)
                d, V = eigh(C)
                d = CovMat._clip(d)
                rk = (d > 0).sum()
                d = d[-rk:][::-1]
                V = (V.T[-rk:][::-1]).T
                self._assign_EVD(M, rk, d, V)
            elif kind == 'diag':
                # With diagonal input, it would be great to use a sparse
                # (or non-existant) representation of V,
                # but that would require so much other adaption of other code.
                d = exactly_1d(data)
                self.diag = d
                M = len(d)
                if np.all(d == d[0]):
                    V = eye(M)
                    rk = M
                else:
                    d = CovMat._clip(d)
                    rk = (d > 0).sum()
                    idx = np.argsort(d)[::-1]
                    d = d[idx][:rk]
                    nn0 = idx < rk
                    V = zeros((M, rk))
                    V[nn0, idx[nn0]] = 1
                self._assign_EVD(M, rk, d, V)
            else:
                raise KeyError

        self._kind = kind
        self._trunc = trunc

    ##################################
    # Protected
    ##################################
    @property
    def M(self):
        """ndims"""
        return self._m

    @property
    def kind(self):
        """Form in which matrix was specified."""
        return self._kind

    @property
    def trunc(self):
        """Truncation threshold."""
        return self._trunc

    ##################################
    # "Non-EVD" stuff
    ##################################
    @property
    def full(self):
        "Full covariance matrix"
        if hasattr(self, '_C'):
            return self._C
        else:
            C = self.Left @ self.Left.T
        self._C = C
        return C

    @lazy_property
    def diag(self):
        "Diagonal of covariance matrix"
        if hasattr(self, '_C'):
            return diag(self._C)
        else:
            return (self.Left ** 2).sum(axis=1)

    @property
    def Left(self):
        """L such that C = L@L.T. Note that L is typically rectangular, but not triangular,
        and that its width is somewhere betwen the rank and M."""
        if hasattr(self, '_R'):
            return self._R.T
        else:
            return self.V * sqrt(self.ews)

    @property
    def Right(self):
        """R such that C = R.T@R. Note that R is typically rectangular, but not triangular,
        and that its height is somewhere betwen the rank and M."""
        if hasattr(self, '_R'):
            return self._R
        else:
            return self.Left.T

    ##################################
    # EVD stuff
    ##################################
    def _assign_EVD(self, M, rk, d, V):
        self._m = M
        self._d = d
        self._V = V
        self._rk = rk

    @staticmethod
    def _clip(d):
        return np.where(d < 1e-8 * d.max(), 0, d)

    def _do_EVD(self):
        if not self.has_done_EVD():
            V, s, UT = svd0(self._R)
            M = UT.shape[1]
            d = s ** 2
            d = CovMat._clip(d)
            rk = (d > 0).sum()
            d = d[:rk]
            V = UT[:rk].T
            self._assign_EVD(M, rk, d, V)

    def has_done_EVD(self):
        """Whether or not eigenvalue decomposition has been done for matrix."""
        return all([key in vars(self) for key in ['_V', '_d', '_rk']])

    @property
    def ews(self):
        """Eigenvalues. Only outputs the positive values (i.e. len(ews)==rk)."""
        self._do_EVD()
        return self._d

    @property
    def V(self):
        """Eigenvectors, output corresponding to ews."""
        self._do_EVD()
        return self._V

    @property
    def rk(self):
        """Rank, i.e. the number of positive eigenvalues."""
        self._do_EVD()
        return self._rk

    ##################################
    # transform_by properties
    ##################################
    def transform_by(self, fun):
        """Generalize scalar functions to covariance matrices
        (via Taylor expansion).
        """

        r = truncate_rank(self.ews, self.trunc, True)
        V = self.V[:, :r]
        w = self.ews[:r]

        return (V * fun(w)) @ V.T

    @lazy_property
    def sym_sqrt(self):
        "S such that C = S@S (and i.e. S is square). Uses trunc-level."
        return self.transform_by(sqrt)

    @lazy_property
    def sym_sqrt_inv(self):
        "S such that C^{-1} = S@S (and i.e. S is square). Uses trunc-level."
        return self.transform_by(lambda x: 1 / sqrt(x))

    @lazy_property
    def pinv(self):
        "Pseudo-inverse. Uses trunc-level."
        return self.transform_by(lambda x: 1 / x)

    @lazy_property
    def inv(self):
        if self.M != self.rk:
            raise RuntimeError("Matrix is rank deficient, " +
                               "and cannot be inverted. Use .tinv() instead?")
        # Temporarily remove any truncation
        tmp = self.trunc
        self._trunc = 1.0
        # Compute and restore truncation level
        Inv = self.pinv
        self._trunc = tmp
        return Inv

    ##################################
    # __repr__
    ##################################
    def __repr__(self):
        s = "\n    M: " + str(self.M)
        s += "\n kind: " + repr(self.kind)
        s += "\ntrunc: " + str(self.trunc)

        # Rank
        s += "\n   rk: "
        if self.has_done_EVD():
            s += str(self.rk)
        else:
            s += "<=" + str(self.Right.shape[0])

        # Full (as affordable)
        s += "\n full:"
        if hasattr(self, '_C') or np.get_printoptions()['threshold'] > self.M ** 2:
            # We can afford to compute full matrix
            t = "\n" + str(self.full)
        else:
            # Only compute corners of full matrix
            K = np.get_printoptions()['edgeitems']
            s += " (only computing/printing corners)"
            if hasattr(self, '_R'):
                U = self.Left[:K, :]  # Upper
                L = self.Left[-K:, :]  # Lower
            else:
                U = self.V[:K, :] * sqrt(self.ews)
                L = self.V[-K:, :] * sqrt(self.ews)

            # Corners
            NW = U @ U.T
            NE = U @ L.T
            SW = L @ U.T
            SE = L @ L.T

            # Concatenate corners. Fill "cross" between them with nan's
            N = np.hstack([NW, nan * ones((K, 1)), NE])
            S = np.hstack([SW, nan * ones((K, 1)), SE])
            All = np.vstack([N, nan * ones(2 * K + 1), S])

            with printoptions(threshold=0):
                t = "\n" + str(All)

        # Indent all of cov array, and add to s
        s += t.replace("\n", "\n   ")

        # Add diag. Indent array +1 vs cov array
        with printoptions(threshold=0):
            s += "\n diag:\n   " + " " + str(self.diag)

        s = repr_type_and_name(self) + s.replace("\n", "\n  ")
        return s

class GaussRV(RV_with_mean_and_cov):
    """Gaussian (Normal) multivariate random variable."""

    def _sample(self, N):
        R = self.C.Right
        D = randn((N, len(R))) @ R
        return D
    
    
# GPR


def get_GP_list(time, num_model, GP_list):
    new_list = {}
    for i in range(num_model):
        name = 'reg_' + str(time) + '_' + str(i)
        tmp = {name: GP_list.get(name)}
        new_list.update(tmp)
    return new_list


def load_GP(set_DA, bb):
    var = set_DA['obs_sigma']
    path = set_DA['data_saved_path']

    name = "GP_" + str(var) + '_' + str(bb) + ".pickle"
    path = Path(path) / 'GP'
    # path = Path('E:\\data_DA\\') / 'GP'

    file_name = path / name

    with file_name.open('rb') as pkl_file:
        GP_list = pickle.load(pkl_file)

    return GP_list

def kernel_cal_class_old(X, list_input,alpha=1):
    kernel = list_input.kernel
    X_train = list_input.X_train
    y_train_ = list_input.y_train
    K = kernel(X_train)
    K[np.diag_indices_from(K)] += alpha
    K_trans = kernel(X, X_train)
    y_var = kernel.diag(X)
    L = cholesky(K, lower=True)  # linalg.cholesky(K)
    alpha = cho_solve((L, True), y_train_)
    L_inv = solve_triangular(L.T, np.eye(L.shape[0]))  # linalg.solve(L.T, np.eye(L.shape[0]))

    return K_trans, y_var,alpha,L_inv

def kernel_class(X, list_input):
    kernel = list_input.kernel
    y_var = kernel.diag(X)
    X_train = list_input.X_train
    K_trans = kernel(X, X_train)

    alpha = list_input.alpha
    K_inv  = list_input.K_inv
    return K_trans, y_var,alpha,K_inv


def predict_jit_class(X,list_input):
    K_trans, y_var,alpha,L_inv= kernel_class(X, list_input)
    y_mean, y_std = predict_cal_jit(K_trans, y_var,alpha,L_inv)
    return y_mean , y_std


@jit(nopython=True)
def predict_cal_jit(K_trans, y_var,alpha,K_inv):

    y_mean = K_trans.dot(alpha)
    # y_mean = y_train_mean + y_mean  # undo normal.
    y_var -= np.sum(np.dot(K_trans, K_inv) * K_trans, axis=1)

    y_var_negative = y_var < 0
    if np.any(y_var_negative):
        # warnings.warn("Predicted variances smaller than 0. " "Setting those variances to 0.")
        y_var[y_var_negative] = 0.0
    return y_mean, np.sqrt(y_var)


def load_GP_list(path,jj,std=0.1):
    path = path +"\\GP_" + str(std) + '_' + str(jj) + ".pickle"
    # path =path +  "\\GP_" + str(std) + '_' +'t_'+ str(jj) # + ".pickle"
    pickling_on = open(path, "rb")
    GP_list = pickle.load(pickling_on)
    pickling_on.close()
    return GP_list

def load_GP_list_new(path,jj,std=0.1):
    # path = path +"\\GP_" + str(std) + '_' + str(jj) + ".pickle"
    path =path +  "\\GP_" + str(std) + '_' +'t_'+ str(jj)  + ".pickle"
    pickling_on = open(path, "rb")
    GP_list = pickle.load(pickling_on)
    pickling_on.close()
    return GP_list
#

def compare_settings(tmp, tmp1):
    for i, j in zip(tmp.items(), tmp1.items()):
        tt = (i[1] == j[1])
        if isinstance(tt, np.ndarray):
            if tt.all():
                pass
        elif tt:
            pass
        else:
            print('fiest_one:', i, 'second_one:', j)


def check_data(path):
    lili = glob.glob(path + '\\*')
    for jiji, nali in enumerate(lili):
        # path = Path('D:\da_data_new\LPF_GT_loc_5_abs_0.5_0.55_std_0.1')
        path = Path(nali)
        path_absolute = os.path.abspath(path / 'stats_res')
        with open(path_absolute, 'rb') as file:
            kaka = dill.load(file)
        print('number:', jiji,' ', nali)
        print(kaka.rmse.f[0:9000].mean())


class localz:
    _name = 'Localization method'

    def __init__(self, set_DA):
        prm = container(set_DA)

        self.loc_fuc = prm.local_func
        self._m = prm.J
        self.ob_indx = prm.obs_inds
        self._scale = prm.local_scale
        self._p = prm.obs_num
        self._x, self._y = np.mgrid[1:(self._m + 1), 0:1]

        self._base_states = list(zip(self._x.ravel(), self._y.ravel()))
        self._base_ob = []
        for i in self.ob_indx:
            self._base_ob.append(self._base_states[i])

    def coeff_x(self, ii):
        cutoff = 1e-3
        # coeffs = np.zeros(len(tmp))

        self.dis = minkowski_distance(self._base_ob, self._base_states[ii])
        self.dist_periodic = np.zeros(len(self.dis))

        for i in range(len(self.dis)):
            self.dist_periodic[i] = min(self.dis[i], self._m - self.dis[i])
        self.coeffs = self.dist2coeff(self.dist_periodic, self._scale)
        self.inds = np.arange(len(self.dist_periodic))[self.coeffs > cutoff]
        self.coeffs_output = self.coeffs[self.inds]

        return self.inds, self.coeffs_output

    def dist2coeff(self, dists, scale, tag='GC'):
        """Compute coefficients corresponding to a distances."""
        coeffs = np.zeros(dists.shape)

        if tag is None:
            tag = 'GC'

        if tag == 'Gauss':
            R = scale
            coeffs = np.exp(-0.5 * (dists / R) ** 2)
        elif tag == 'Exp':
            R = scale
            coeffs = np.exp(-0.5 * (dists / R) ** 3)
        elif tag == 'Cubic':
            R = scale * 1.8676
            inds = dists <= R
            coeffs[inds] = (1 - (dists[inds] / R) ** 3) ** 3
        elif tag == 'Quadro':
            R = scale * 1.7080
            inds = dists <= R
            coeffs[inds] = (1 - (dists[inds] / R) ** 4) ** 4
        elif tag == 'GC':
            # Gaspari_Cohn
            R = scale * 1.7386  # Sakov: 1.82
            #
            ind1 = dists <= R
            r2 = (dists[ind1] / R) ** 2
            r3 = (dists[ind1] / R) ** 3
            coeffs[ind1] = 1 + r2 * (- r3 / 4 + r2 / 2) + r3 * (5 / 8) - r2 * (5 / 3)
            #
            ind2 = np.logical_and(R < dists, dists <= 2 * R)
            r1 = (dists[ind2] / R)
            r2 = (dists[ind2] / R) ** 2
            r3 = (dists[ind2] / R) ** 3
            coeffs[ind2] = r2 * (r3 / 12 - r2 / 2) + r3 * (5 / 8) + r2 * (5 / 3) - r1 * 5 + 4 - (2 / 3) / r1
        elif tag == 'Step':
            R = scale
            inds = dists <= R
            coeffs[inds] = 1
        else:
            raise KeyError('No such coeff function.')

        return coeffs

    def coeff_y(self, y_indx):
        self.dis = minkowski_distance(self._base_ob[y_indx], self._base_states)
        self.dist_periodic = np.zeros(len(self.dis))
        for i in range(len(self.dis)):
            self.dist_periodic[i] = min(self.dis[i], self._m - self.dis[i])
        self.coeffs = self.dist2coeff(self.dist_periodic, self._scale)

        return self.coeffs




def outliers(set_DA,ens,ens_initial):

    truth_max = 50
    truth_min = 50 *(-1)

    ens_domain_min = 50 *(-1)
    ens_domain_max = 50


    ens_total_tmp = ens
    mean1 = ens_total_tmp.mean(axis=0)

    # idx1_1 = np.argwhere(mean1 > 1000)
    # idx2_1 = np.argwhere(mean1 < -1000)
    #
    # if idx2_1.size + idx1_1.size > 0:
    #     print(mean1)
    #     print(idx1_1)
    #     print(idx2_1)
    # print('final update - Time:' +str(time))
    # break

    ####################################################################################################################

    tt = 0

    idx1 = np.argwhere(mean1 > ens_domain_max)
    idx2 = np.argwhere(mean1 < ens_domain_min)
    idx3 = np.argwhere(ens_total_tmp > truth_max)
    idx4 = np.argwhere(ens_total_tmp < truth_min)

    # print('idx1: '+str(idx1)+"__"+'idx2: '+str(idx2)+"__"+'idx3: '+str(idx3)+"__"+'idx4: '+str(idx4),file=prm._file_txt)

    while (idx1.size + idx2.size + idx3.size + idx4.size) > 0:
        check_outliers_class = check_outliers(set_DA,ens_forward=ens_total_tmp, ens_initial=ens_initial,
                                              ens_domain_min=ens_domain_min, ens_domain_max=ens_domain_max,
                                              x_min=truth_min, x_max=truth_max)
        ens_total_tmp = check_outliers_class.check()

        mean1 = ens_total_tmp.mean(axis=0)
        idx1 = np.argwhere(mean1 > ens_domain_max)
        idx2 = np.argwhere(mean1 < ens_domain_min)
        idx3 = np.argwhere(ens_total_tmp > truth_max)
        idx4 = np.argwhere(ens_total_tmp < truth_min)
        tt = tt + 1
        # print("outliers remove:" + str(tt) + ' times',file=prm._file_txt)

    return  ens_total_tmp

class check_outliers:

    def __init__(self,set_DA,ens_forward=None,ens_initial=None,ens_domain_min=None,ens_domain_max=None,x_min = None,x_max = None):
        self.ens_forward = ens_forward
        self.ens_domain_max =ens_domain_max
        self.ens_domain_min =ens_domain_min
        self.ens_initial =ens_initial
        self.n,self.m = ens_forward.shape
        self.x_min = x_min
        self.x_max = x_max

    def check(self):

        self.ens_mean = self.ens_forward.mean(axis=0)

        #index_mean_max,index_mean_min = self.find_index(ens_mean,self.ens_domain_max,self.ens_domain_min)
        self.index_mean_max, self.index_mean_min = self.find_index(self.ens_mean, self.ens_domain_max, self.ens_domain_min)

        self.index_mean_max = self.index_mean_max.ravel()
        self.index_mean_min = self.index_mean_min.ravel()

        # print("--------------------index_mean_max--------------------",file=self.file_txt)
        # print(self.index_mean_max,file=self.file_txt)
        #
        # print("--------------------index_mean_min--------------------",file=self.file_txt)
        # print(self.index_mean_min,file=self.file_txt)

        if self.index_mean_max.size > 0:
            for i in self.index_mean_max:
                for j in range(self.n):
                    if self.ens_forward[j,i] > self.ens_domain_max:
                        self.ens_forward[j,i] = self.ens_initial[j,i]

        if self.index_mean_min.size > 0:
            for i in self.index_mean_min:
                for j in range(self.n):
                    if self.ens_forward[j,i] < self.ens_domain_min:
                        self.ens_forward[j,i] = self.ens_initial[j,i]

        self.ens_max = self.x_max
        self.ens_min = self.x_min

        self.index_max, self.index_min = self.find_index(self.ens_forward, self.ens_max,self.ens_min)

        self.index_max_2nd = self.index_max[:,1].ravel()
        self.index_min_2nd = self.index_min[:,1].ravel()

        if self.index_max.size > 0:
            for i in self.index_max_2nd:
                for j in range(self.n):
                    if self.ens_forward[j,i] > self.ens_max:
                        self.ens_forward[j,i] = self.ens_initial[j,i]

        if self.index_min.size > 0:
            for i in self.index_min_2nd:
                for j in range(self.n):
                    if self.ens_forward[j,i] < self.ens_min:
                        self.ens_forward[j,i] = self.ens_initial[j,i]

        return self.ens_forward

    def find_index(self,ens,max,min):
        index_max = np.argwhere(ens > max)
        index_min = np.argwhere(ens < min)
        return  index_max,index_min

    def replace_outliers(self):
        pass