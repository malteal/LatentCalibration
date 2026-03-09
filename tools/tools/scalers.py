"Normalize data"

import numpy as np
from sklearn.preprocessing import MinMaxScaler, minmax_scale, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
import joblib
import sys

def norm_type(name:str, args:dict={}):
    if name.casefold() == "quantile":
        scaler = QuantileTransformer(**args)
    elif name.casefold() == "robust":
        scaler = RobustScaler(**args)
    elif name.casefold() == "standard":
        scaler = StandardScaler(**args)
    elif np.isin(name.casefold(), ["minmax", 'uniform']):
        scaler = MinMaxScaler(**args)
    else:
        raise ValueError("Scaler unknown")

    return scaler

def load_scaler(scaler_path: str):
    "load scaler"
    scaler = joblib.load(scaler_path) 
    return scaler

def save_scaler(scaler, save_path: str):
    "save scaler"
    joblib.dump(scaler, save_path) 

def run_scaler(name, data=None, scaler_attr:dict={}, load_scaler_path: str = None,
               save_scaler_path:str=None, inverse:bool=False):

    assert not (save_scaler_path is not None and load_scaler_path is not None), "Cannot overload saved scaler"

    scaler = norm_type(name, scaler_attr)

    if load_scaler_path is not None:
        scaler = load_scaler(scaler, load_scaler_path)
    
    if data is None:
        return scaler

    if inverse:
        norm_data = scaler.inverse_transform(data)
    else:
        if load_scaler_path is None:
            norm_data = scaler.fit_transform(data)
        else:
            norm_data = scaler.transform(data)
        
        if save_scaler_path is not None:
            save_scaler(scaler, save_scaler_path)
    return scaler, norm_data
