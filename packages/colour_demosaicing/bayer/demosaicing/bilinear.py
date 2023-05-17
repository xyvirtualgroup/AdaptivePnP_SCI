# -*- coding: utf-8 -*-
"""
Bilinear Bayer CFA Demosaicing
==============================

*Bayer* CFA (Colour Filter Array) bilinear demosaicing.

References
----------
-   :cite:`Losson2010c` : Losson, O., Macaire, L., & Yang, Y. (2010).
    Comparison of Color Demosaicing Methods. In Advances in Imaging and
    Electron Physics (Vol. 162, pp. 173-265). doi:10.1016/S1076-5670(10)62005-8
"""

from __future__ import division, unicode_literals
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.filters import convolve

from colour.utilities import as_float_array, tstack

from packages.colour_demosaicing.bayer import masks_CFA_Bayer

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['demosaicing_CFA_Bayer_bilinear']


def demosaicing_CFA_Bayer_bilinear(CFA, pattern='RGGB'):
    """
    Returns the demosaiced *RGB* colourspace array from given *Bayer* CFA using
    bilinear interpolation.

    Parameters
    ----------
    CFA : array_like
        *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   The definition output is not clipped in range [0, 1] : this allows for
        direct HDRI / radiance image generation on *Bayer* CFA data and post
        demosaicing of the high dynamic range data as showcased in this
        `Jupyter Notebook <https://github.com/colour-science/colour-hdri/\
blob/develop/colour_hdri/examples/\
examples_merge_from_raw_files_with_post_demosaicing.ipynb>`__.

    References
    ----------
    :cite:`Losson2010c`

    Examples
    --------
    >>> import numpy as np
    >>> CFA = np.array(
    ...     [[0.30980393, 0.36078432, 0.30588236, 0.3764706],
    ...      [0.35686275, 0.39607844, 0.36078432, 0.40000001]])
    >>> demosaicing_CFA_Bayer_bilinear(CFA)
    array([[[ 0.69705884,  0.17941177,  0.09901961],
            [ 0.46176472,  0.4509804 ,  0.19803922],
            [ 0.45882354,  0.27450981,  0.19901961],
            [ 0.22941177,  0.5647059 ,  0.30000001]],
    <BLANKLINE>
           [[ 0.23235295,  0.53529412,  0.29705883],
            [ 0.15392157,  0.26960785,  0.59411766],
            [ 0.15294118,  0.4509804 ,  0.59705884],
            [ 0.07647059,  0.18431373,  0.90000002]]])
    >>> CFA = np.array(
    ...     [[0.3764706, 0.360784320, 0.40784314, 0.3764706],
    ...      [0.35686275, 0.30980393, 0.36078432, 0.29803923]])
    >>> demosaicing_CFA_Bayer_bilinear(CFA, 'BGGR')
    array([[[ 0.07745098,  0.17941177,  0.84705885],
            [ 0.15490197,  0.4509804 ,  0.5882353 ],
            [ 0.15196079,  0.27450981,  0.61176471],
            [ 0.22352942,  0.5647059 ,  0.30588235]],
    <BLANKLINE>
           [[ 0.23235295,  0.53529412,  0.28235295],
            [ 0.4647059 ,  0.26960785,  0.19607843],
            [ 0.45588237,  0.4509804 ,  0.20392157],
            [ 0.67058827,  0.18431373,  0.10196078]]])
    """

    CFA = as_float_array(CFA)
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    H_G = as_float_array(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]]) / 4  # yapf: disable

    H_RB = as_float_array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]) / 4  # yapf: disable

    R = convolve(CFA * R_m, H_RB)
    G = convolve(CFA * G_m, H_G)
    B = convolve(CFA * B_m, H_RB)

    del R_m, G_m, B_m, H_RB, H_G

    return tstack([R, G, B])

def np2tch_cuda(a):
    return torch.from_numpy(a).cuda()
def cuda2np(a):
    return a.cpu().detach().numpy()
def as_array(a, dtype=None):
    return torch.tensor(a, dtype=torch.float64).cuda()


def demosaicing_CFA_Bayer_bilinear_tensor(CFA,R_m, G_m, B_m, pattern='RGGB'):

    CFA = CFA.cuda()
    # CFA_np = cuda2np(CFA)

    H_G = as_array(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]]) / 4  # yapf: disable

    H_RB = as_array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]) / 4  # yapf: disable

    # GR_GB = GR_GB.unsqueeze(0).unsqueeze(0).float()
    H_RB = H_RB.unsqueeze(0).unsqueeze(0).float()
    H_G = H_G.unsqueeze(0).unsqueeze(0).float()

    
    CFA_R = CFA * R_m
    CFA_G = CFA * G_m
    CFA_B = CFA * B_m

    CFA_R = CFA_R.unsqueeze(0).unsqueeze(0)
    CFA_G = CFA_G.unsqueeze(0).unsqueeze(0)
    CFA_B = CFA_B.unsqueeze(0).unsqueeze(0)
    # R = convolve(CFA * R_m, H_RB)
    # G = convolve(CFA * G_m, H_G)
    # B = convolve(CFA * B_m, H_RB)
    CFA_R = F.pad(CFA_R,(1,1,1,1))
    CFA_G = F.pad(CFA_G,(1,1,1,1))
    CFA_B = F.pad(CFA_B,(1,1,1,1))

    R = F.conv2d(CFA_R, H_RB,stride=1).squeeze(0).squeeze(0)
    G = F.conv2d(CFA_G, H_G,stride=1).squeeze(0).squeeze(0)
    B = F.conv2d(CFA_B, H_RB,stride=1).squeeze(0).squeeze(0)

    del R_m, G_m, B_m, H_RB, H_G
    return torch.cat([x.unsqueeze(2) for x in [R, G, B]], axis=-1)
    # return tstack([R, G, B])