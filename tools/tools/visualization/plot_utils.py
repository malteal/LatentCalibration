# Description: This file contains utility functions for plotting data

from copy import deepcopy
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def bold(string) -> str:
    return f"$\\bf{string.replace('$', '')}$"

def italic(string) -> str:
    return f"$\\it{string.replace('$', '')}$"

def italic_bold(string) -> str:
    return f"$\\bf{string}$"

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def merge_bins(bins:np.ndarray, counts:np.ndarray, threshold:int) -> tuple:
    merged_bins = deepcopy(bins)
    merged_counts = deepcopy(counts)

    for add_to_right in [True, False]:

        i = 0 if add_to_right else len(merged_counts) - 1
        
        # flip boolean for left cumulation to right cumulation
        boolean= lambda x, m_c: x < len(m_c) - 1 if add_to_right else x > 0
        
        # boolean to check if we should break out of the while loop
        breakout = lambda skip, m_c, i: skip > len(m_c) - 1 if add_to_right else (skip < 0) | (i == len(m_c))

        while boolean(i, merged_counts):
            # If the count is below the threshold, merge it with the next bin
            if add_to_right:
                skip = i+1
            else:
                skip = i-1

            while merged_counts[i] < threshold:
                # If the count is below the threshold, merge it with the next bin on the right
                merged_counts[i] = merged_counts[i] + merged_counts[skip]

                # delete counts on the right
                merged_counts = np.delete(merged_counts, skip)
                
                # delete bins on the right
                merged_bins = np.delete(merged_bins, skip)

                if breakout(skip, merged_counts, i):
                    break

            if add_to_right:
                i +=1
            else:
                i -=1
    if np.sum(merged_counts) != np.sum(counts):
        raise ValueError("The sum of the bins has changed after merging")
    
    # if (merged_bins[0] != bins[0]) | (merged_bins[-1] != bins[-1]):
    #     raise ValueError("The bin edges have changed after merging")

    return merged_bins, merged_counts

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
