# Image transformations for optimizing raster plots.

import numpy as np
################################################################
# FORWARD SCALINGS

# Scaling transformation if power law index, gamma < 0.
def scale_image_negative_powerlawindex(image,powerlaw_index,vmin,vmax):
    # Mask values (including NaNs) less than vmin.
    masked_image = np.ma.masked_less(np.nan_to_num(image,nan=vmin-2),vmin)
    # Mask values greater than vmax.
    masked_image = np.ma.masked_greater(masked_image,vmax)
    # Scale [vmin, vmax] linearly to [1, 10**p].  Here, (-p) is the power-law index.
    p = -powerlaw_index
    y = 1.0 + (10**p - 1)*(masked_image-vmin)/(vmax-vmin)
    # Take log_10 to restrict output values to range [0, p].
    Y = np.log10(y)
    return Y

# Scaling transformation if power law index, gamma > 0.
def scale_image_positive_powerlawindex(image,powerlaw_index,vmin,vmax):
    # Mask values (including NaNs) less than vmin.
    masked_image = np.ma.masked_less(np.nan_to_num(image,nan=vmin-2),vmin)
    # Mask values greater than vmax.
    masked_image = np.ma.masked_greater(masked_image,vmax)
    # Scale [vmin, vmax] linearly to [0, p]. Here, p is the power-law index.
    y = powerlaw_index*(masked_image-vmin)/(vmax-vmin)
    Y = 10**y
    return Y

# Grand function that scales images for plotting according to the value of powerlaw index (gamma) specified.
def scale_image(image,gamma,vmin,vmax):
    if gamma<0:
        Y = scale_image_negative_powerlawindex(image,gamma,vmin,vmax)
    elif gamma>0:
        Y = scale_image_positive_powerlawindex(image,gamma,vmin,vmax)
    else:
        # Mask values (including NaNs) less than vmin.
        Y = np.ma.masked_less(np.nan_to_num(image,nan=vmin-2),vmin)
        # Mask values greater than vmax.
        Y = np.ma.masked_greater(Y,vmax)
    return Y
################################################################
# INVERSE TRANFORMATIONS

# Inverse transform an image using a supplied negative powerlaw index.
def inverse_transform_negative_powerlawindex(image,powerlaw_index,vmin,vmax):
    p = -powerlaw_index
    y = 10**image
    x = (y-1)*(vmax-vmin)/(10**p - 1) + vmin
    return x

# Inverse transform an image using a supplied positive powerlaw index.
def inverse_transform_positive_powerlawindex(image,powerlaw_index,vmin,vmax):
    y = np.log10(image)
    x = y*(vmax-vmin)/powerlaw_index + vmin
    return x

# Grand function that inverse transforms an image using a specified powerlaw index (gamma).
def inverse_transform_image(image,gamma,vmin,vmax):
    if gamma<0:
        x = inverse_transform_negative_powerlawindex(image,gamma,vmin,vmax)
    elif gamma>0:
        x = inverse_transform_positive_powerlawindex(image,gamma,vmin,vmax)
    else:
        x = image
    return x
################################################################
