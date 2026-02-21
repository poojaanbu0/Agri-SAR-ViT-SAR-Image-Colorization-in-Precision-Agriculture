import numpy as np
from scipy.ndimage import uniform_filter

def radiometric_calibration(band_data):
    """
    Transforms raw intensity values into physically meaningful 
    backscatter coefficients (Sigma Naught).
    """
    # Typically involves converting to decibels (dB)
    # Sigma0_dB = 10 * log10(abs(intensity))
    return 10 * np.log10(np.abs(band_data) + 1e-10)

def lee_filter(img, size=5):
    """
    Standard adaptive Lee filter for speckle reduction while 
    maintaining structural boundaries.
    """
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance + 1e-10)
    
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def preprocess_sar(vv_band, vh_band):
    """
    Complete pipeline for dual-polarized (VV, VH) data.
    """
    vv_cal = radiometric_calibration(vv_band)
    vh_cal = radiometric_calibration(vh_band)
    
    vv_filtered = lee_filter(vv_cal)
    vh_filtered = lee_filter(vh_cal)
    
    return vv_filtered, vh_filtered
