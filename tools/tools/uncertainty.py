
import numpy as np


def divuncorr(xs:np.ndarray, ys:np.ndarray, dxs:np.ndarray, dys:np.ndarray):
    """
    fractional uncertainty of the division of two uncorrelated variables
    
    xs: np.ndarray - fractions of the numerator
    ys: np.ndarray - fractions of the denominator
    dxs: uncertainty of fraction
    dys: uncertainty of fraction
    """

    zs = np.where(ys == 0.0, 0.0, xs / ys)
    dzs = np.sqrt(dxs**2 + (zs * dys)**2) / ys
    return zs , dzs

def weighted_bins_digitized(data:np.ndarray, selection: np.ndarray,
                            bins:np.ndarray, weights:np.ndarray) -> list:
    
    inds = np.digitize(data, bins, right=False)

    unc_lst = []
    for ind in np.unique(inds):
        mask = inds == ind
        unc_lst.append(weighted_binomial(weights[mask], selection[mask]))
        
    return unc_lst
    

def weighted_binomial(weights:np.ndarray, selection:np.ndarray):
    """
    Calculate the weighted binomial uncertainty.
    from: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Standard_error_of_a_proportion_estimation_when_using_weighted_data

    Parameters:
    weights (numpy.ndarray): An array of weights after selection and normalised.

    Returns:
    float: The calculated weighted binomial uncertainty.
    """
    if weights.sum() > 1:
        weights = weights/weights.sum()

    # ratio that pass the working point
    p_hat = weights[selection].sum()
    
    # uncertainty of the weighted binomial
    unc = np.sqrt(p_hat*(1-p_hat) * (weights**2).sum())

    return unc

def binomial(counts:np.ndarray, total=None, normalise:bool=True) -> np.ndarray:
    """
    Calculate the binomial uncertainty for a set of counts.
    
    counts: np.ndarray - Should be integers for the ratio below
    """
    #https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf
    #https://inspirehep.net/files/57287ac8e45a976ab423f3dd456af694
    
    total = np.sum(counts) if total is None else total
    p = counts/total

    if normalise:
        return np.sqrt(p*(1-p)/total)
    else:
        return np.sqrt(p*(1-p)*total)