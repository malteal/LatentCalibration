"calculate relative position to the jet and jet variables"

import copy
import numpy as np
from typing import List, Dict, Union
import torch as T
import pandas as pd
from dataclasses import dataclass, field

from ..transformations import log_squash, undo_log_squash

JET_COL = ["px","py","pz","eta", "phi", "pt", "mass"]

@dataclass()
class LorentzVector:
    # [{"eta": float, "phi": float, "pt": float}]
    jets: List[Dict[str, float]]
    columns:list = field(init=True, default_factory=lambda: ["eta", "phi", "pt"])
    
    # internal
    _vec_4_mom_dict:dict = None
    _deltaR:float = None
    _sumpt:float = None
    _eta:float = None
    _phi:float = None

    def __post_init__(self) -> None:
        
        if not np.all([isinstance(i, dict) for i in self.jets]):
            raise TypeError("All jets have to be dict")
        
        if not np.all([np.in1d(self.columns, list(i.keys())) for i in self.jets]):
            for i in range(len(self.jets)):
                self.jets[i]["pt"] = calculate_pT(self.jets[i]["px"], self.jets[i]["py"])
                self.jets[i]["eta"] = calculate_eta(self.jets[i]["pz"], self.jets[i]["pt"])
                self.jets[i]["phi"] = calculate_phi(self.jets[i]["px"], self.jets[i]["py"])
            
            
        
        if not np.all([np.in1d(self.columns, list(i.keys()))
            for i in self.jets]):
            raise ValueError("Columns are missing in the jet")
         
        if len(self.jets)==2:
            self._deltaR = deltaR(self.jets[0]["eta"],self.jets[0]["phi"],
                                  self.jets[1]["eta"],self.jets[1]["phi"])[0]
        else:
            self._deltaR=None

        self._sumpt=np.sum([self.jets[i]["pt"] for i in range(len(self.jets))])

        self._vec_4_mom_dict = calculate_mass_of_jet(self.jets)
        
        self._eta = calculate_eta(self._vec_4_mom_dict["pz"], self._vec_4_mom_dict["pt"])

        self._phi = calculate_phi(self._vec_4_mom_dict["px"], self._vec_4_mom_dict["py"])
    
    def __add__(self, jets):
        if isinstance(jets, LorentzVector):
            return LorentzVector(self.jets+jets.jets)
        elif isinstance(jets, list):
            return LorentzVector(self.jets+jets)
        
    # get jet values
    def M(self):
        return self._vec_4_mom_dict["mass"]

    def deltaR(self):
        return self._deltaR

    def eta(self):
        return  self._eta
    
    def phi(self):
        return  self._phi

    def pt(self):
        return self._vec_4_mom_dict["pt"]

    def px(self):
        return self._vec_4_mom_dict["px"]

    def py(self):
        return self._vec_4_mom_dict["py"]

    def pz(self):
        return self._vec_4_mom_dict["pz"]

    def energy(self):
        return self._vec_4_mom_dict["e"]

    def sumpt(self):
        return self._sumpt

def rescale_phi(phi):
    phi[phi >= np.pi] -= 2*np.pi
    phi[phi < -np.pi] += 2*np.pi
    return phi

def deltaR(eta1:float, phi1:float, eta2:float, phi2:float) -> np.ndarray:
    deta = (eta1-eta2)
    dphi = rescale_phi(np.array([phi1-phi2]))
    return np.sqrt(deta**2+dphi**2)

def relative_pos(cnts_vars, jet_vars, mask, reverse=False, pt_trans=""):
    "Calculate relative position to the jet"
    cnts_vars_rel = copy.deepcopy(cnts_vars)

    if isinstance(jet_vars, pd.DataFrame):
        jet_vars = jet_vars[["eta", "phi", "pt"]].values
    
    if reverse:
        # if isinstance(jet_vars, pd.DataFrame):
        #     raise NotImplemented("jet_vars should be a numpy array - should implement pandas")

        # from relative position to abs position
        for nr, i in enumerate(['eta', 'phi']):
            cnts_vars_rel[..., nr] += jet_vars[:, nr][:,None]
            if i in "phi":
                cnts_vars_rel[..., nr] = rescale_phi(cnts_vars_rel[..., nr])
        if "ratio" in pt_trans:
            cnts_vars_rel[..., nr+1] = cnts_vars_rel[..., nr+1]*jet_vars[:, nr+1][:,None]
        elif "log_squash" in pt_trans:
            cnts_vars_rel[..., nr+1] = undo_log_squash(cnts_vars_rel[..., nr+1])
        else:
            cnts_vars_rel[..., nr+1] = np.exp(cnts_vars_rel[..., nr+1])

    else:
        # from abs position to relative position
        for nr, i in enumerate(['eta', 'phi']):
            cnts_vars_rel[..., nr] -= jet_vars[:, nr][:,None]
            if i in "phi":
                cnts_vars_rel[..., nr] = rescale_phi(cnts_vars_rel[..., nr])


        # log squash pT
        if pt_trans == "log_squash":
            cnts_vars_rel[..., nr+1] = log_squash(cnts_vars_rel[..., nr+1])
        elif pt_trans == "log":
            cnts_vars_rel[..., nr+1][mask] = np.clip(cnts_vars_rel[..., nr+1][mask],
                                                        a_min=1e-8, a_max=None)
            cnts_vars_rel[..., nr+1] = np.log(cnts_vars_rel[..., nr+1])
        else:
            raise ValueError("pt_trans should be log_squash or log")
        
        cnts_vars_rel[..., nr+1] = np.nan_to_num(cnts_vars_rel[..., nr+1], -1)


    if mask is not None:
        cnts_vars_rel[~mask]=0
    
    return cnts_vars_rel 


def calculate_pT(px, py):
    return np.sqrt(px**2+py**2)

def calculate_phi(px, py):
    return np.arctan2(py, px) # np.arccos(px/pT)

def calculate_eta(pz, pT):
    return np.arcsinh(pz/pT)

def detector_dimensions(df:np.ndarray):

    # calculate pT
    pT = calculate_pT(df[..., 0], df[..., 1])

    # calculate phi
    phi = calculate_phi(df[..., 0], df[..., 1])

    # calculate eta
    eta = calculate_eta(df[..., 2], pT)
    
    return eta[..., None], phi[..., None], pT[..., None]

def jet_variables(sample, mask):
    
    # calculate summary jet features
    jet_vars = numpy_locals_to_mass_and_pt(sample, mask)
    
    # into a df
    jet_vars = pd.DataFrame(jet_vars, columns=JET_COL)

    return jet_vars

def torch_locals_to_mass_and_pt(
    csts: T.Tensor,
    mask: T.BoolTensor,
    undo_logsquash: bool = False,
) -> T.Tensor:
    """Calculate the overall jet pt and mass from the constituents. The
    constituents are expected to be expressed as:

    - eta
    - phi
    - pt or log_squash_pt
    """

    # Calculate the constituent pt, eta and phi
    eta = csts[..., 0]
    phi = csts[..., 1]
    pt = undo_log_squash(csts[..., 2]) if undo_logsquash else csts[..., 2]

    # Calculate the total jet values in cartensian coordinates, include mask for sum
    jet_px = (pt * T.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * T.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * T.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * T.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = T.clamp_min(jet_px**2 + jet_py**2, 0).sqrt()
    jet_m = T.clamp_min(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0).sqrt()

    return T.vstack([jet_pt, jet_m]).T

def calculate_mass_of_jet(jets: Union[Dict, np.ndarray],
                          axis):
    """Calculate the overall jet pt and mass from the constituents. 
    IMPORTANT ensure order is pt, eta, phi in np.array
    """
    if isinstance(jets, dict):
        pt = [jets[i]["pt"] for i in range(len(jets))]
        eta = [jets[i]["eta"] for i in range(len(jets))]
        phi = [jets[i]["phi"] for i in range(len(jets))]
    else:
        pt = jets[..., 0]
        eta = jets[..., 1]
        phi = jets[..., 2]
        
        # energy = [jets[i]["energy"] for i in range(len(jets))]
    jet_px = (pt * np.cos(phi))
    jet_py = (pt * np.sin(phi))
    jet_pz = (pt * np.sinh(eta))
    jet_e = (pt * np.cosh(eta))

    if axis is not None:
        jet_px = jet_px.sum(axis)
        jet_py = jet_py.sum(axis)
        jet_pz = jet_pz.sum(axis)
        jet_e = jet_e.sum(axis)
            

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = np.sqrt(np.clip(jet_px**2 + jet_py**2, 0, None))
    jet_m = np.sqrt(np.clip(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0, None))

    return {"px":jet_px,"py": jet_py, "pz":jet_pz, "pt":jet_pt,"e": jet_e,"mass": jet_m}

def numpy_locals_to_mass_and_pt(csts: np.ndarray, mask: np.ndarray,
                                undo_logsquash: bool = False
                                ) -> np.ndarray:
    """Calculate the overall jet pt and mass from the constituents. The
    constituents are expected to be expressed as:

    - eta
    - phi
    - pt
    """

    # Calculate the constituent pt, eta and phi
    eta = csts[..., 0]
    phi = csts[..., 1]
    pt = undo_log_squash(csts[..., 2]) if undo_logsquash else csts[..., 2]
    

    # Calculate the total jet values in cartensian coordinates, include mask for sum
    if len(mask.shape)==1:
        jet_px = np.sum(pt[mask] * np.cos(phi[mask]), axis=-1)
        jet_py = np.sum(pt[mask] * np.sin(phi[mask]), axis=-1)
        jet_pz = np.sum(pt[mask] * np.sinh(eta[mask]), axis=-1)
        jet_e = np.sum(pt[mask] * np.cosh(eta[mask]), axis=-1)
    else:
        jet_px = np.sum(pt * np.cos(phi) * mask, axis=-1)
        jet_py = np.sum(pt * np.sin(phi) * mask, axis=-1)
        jet_pz = np.sum(pt * np.sinh(eta) * mask, axis=-1)
        jet_e = np.sum(pt * np.cosh(eta) * mask, axis=-1)
    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = np.sqrt(np.clip(jet_px**2 + jet_py**2, 0, None))
    jet_m = np.sqrt(np.clip(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0, None))
    
    # position of jet
    jet_phi = calculate_phi(jet_px, jet_py)
    jet_eta = calculate_eta(jet_pz, jet_pt)

    return np.vstack([jet_px, jet_py, jet_pz, jet_eta, jet_phi, jet_pt, jet_m]).T



if __name__ == "__main__":
    pass