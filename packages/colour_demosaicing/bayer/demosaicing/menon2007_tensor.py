# -*- coding: utf-8 -*-
"""
DDFAPD - Menon (2007) Bayer CFA Demosaicing
===========================================

*Bayer* CFA (Colour Filter Array) DDFAPD - *Menon (2007)* demosaicing.

References
----------
-   :cite:`Menon2007c` : Menon, D., Andriani, S., & Calvagno, G. (2007).
    Demosaicing With Directional Filtering and a posteriori Decision. IEEE
    Transactions on Image Processing, 16(1), 132-141.
    doi:10.1109/TIP.2006.884928
"""

from __future__ import division, unicode_literals

import numpy as np
from numpy.lib.arraypad import pad
# from scipy.ndimage.filters import convolve, convolve1d

# from colour.utilities import as_float_array, tsplit, tstack
import torch
from torch.nn import functional as F
from packages.colour_demosaicing.bayer import masks_CFA_Bayer

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'demosaicing_CFA_Bayer_Menon2007', 'demosaicing_CFA_Bayer_DDFAPD',
    'refining_step_Menon2007'
]


def _cnv_h(x, y):
    """
    Helper function for horizontal convolution.
    """
    x = x.reshape(1,512*512).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(0)
    cnv_h = F.conv1d(x, y,stride=(1),padding=2).squeeze(0).reshape(512,512)
    return  cnv_h #convolve1d(x, y, mode='mirror')


def _cnv_v(x, y):
    """
    Helper function for vertical convolution.
    """
    x = x.permute(1,0)
    x = x.reshape(1,512*512).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(0)
    conv_v = F.conv1d(x, y,stride=(1),padding=2).squeeze(0).reshape(512,512).permute(1,0)
    return conv_v#convolve1d(x, y, mode='mirror', axis=0)

def np2tch_cuda(a):
    return torch.from_numpy(a).cuda()
def cuda2np(a):
    return a.cpu().detach().numpy()
def as_array(a, dtype=None):
    return torch.tensor(a,dtype=torch.float)

def as_float_array(a, dtype=None):
    return as_array(a, dtype)

def tstack(list):
    return torch.cat([x.unsqueeze(2) for x in list], axis=-1)
def tsplit(x):
    return x[0],x[1],x[2]

def convolve(x,w,mode):
    x = x.unsqueeze(0).unsqueeze(0)
    w = w.unsqueeze(0).unsqueeze(0)
    return F.conv2d(x,w,padding=2).squeeze(0).squeeze(0)


def demosaicing_CFA_Bayer_Menon2007(CFA, R_m, G_m, B_m,pattern='RGGB', refining_step=True,):
   

    CFA = as_float_array(CFA)
    # R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    h_0 = torch.tensor([0, 0.5, 0, 0.5, 0]).cuda()
    h_1 = torch.tensor([-0.25, 0, 0.5, 0, -0.25]).cuda()

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    convh0 = _cnv_h(CFA, h_0)
    convh1 = _cnv_h(CFA, h_1)

    convv0 = _cnv_v(CFA, h_0) 
    convv1 = _cnv_v(CFA, h_1)

    G_H = torch.where(G_m == 0, convh0 +convh1 , G)
    G_V = torch.where(G_m == 0, convv0+convv1 , G)
    zero = torch.tensor(0).cuda().float()


    C_H = torch.where(R_m == 1, R - G_H, zero )
    C_H = torch.where(B_m == 1, B - G_H, C_H)

    C_V = torch.where(R_m == 1, R - G_V, zero)
    C_V = torch.where(B_m == 1, B - G_V, C_V)
    # torch.constant_pad_nd(x,)
    pad1 = F.pad(C_H.unsqueeze(0), ((0, 0),(0, 0), (0, 2)), mode=str('reflect')).squeeze(0)

    pad2 = F.pad(C_V.unsqueeze(0), ((0, 0),(0, 2),(0, 0)), mode=str('reflect')).squeeze(0)

    D_H = torch.abs(C_H - pad1[:, 2:])
    D_V = torch.abs(C_V - pad2[2:, :])

    del h_0, h_1, CFA, C_V, C_H

    k = torch.tensor(
        [[0, 0, 1, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 3, 0, 3],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 0, 1]]).cuda()  # yapf: disable

    d_H = convolve(D_H, k, mode='constant')
    d_V = convolve(D_V, (k).permute(1,0), mode='constant')

    del D_H, D_V

    mask = d_V >= d_H
    G = torch.where(mask, G_H, G_V)
    M = torch.where(mask, 1, 0)

    del d_H, d_V, G_H, G_V

    # Red rows.
    R_r = (torch.any(R_m == 1, axis=1).unsqueeze(0)).permute(1,0) * torch.ones(R.shape).cuda()
    # Blue rows.
    B_r = (torch.any(B_m == 1, axis=1).unsqueeze(0)).permute(1,0) * torch.ones(B.shape).cuda()

    k_b = torch.tensor([0.5, 0, 0.5]).cuda()

    R = torch.where(
        torch.logical_and(G_m == 1, R_r == 1),
        G + _cnv_h(R, k_b) - _cnv_h(G, k_b),
        R,
    )

    R = torch.where(
        torch.logical_and(G_m == 1, B_r == 1) == 1,
        G + _cnv_v(R, k_b) - _cnv_v(G, k_b),
        R,
    )

    B = torch.where(
        torch.logical_and(G_m == 1, B_r == 1),
        G + _cnv_h(B, k_b) - _cnv_h(G, k_b),
        B,
    )

    B = torch.where(
        torch.logical_and(G_m == 1, R_r == 1) == 1,
        G + _cnv_v(B, k_b) - _cnv_v(G, k_b),
        B,
    )

    R = torch.where(
        torch.logical_and(B_r == 1, B_m == 1),
        torch.where(
            M == 1,
            B + _cnv_h(R, k_b) - _cnv_h(B, k_b),
            B + _cnv_v(R, k_b) - _cnv_v(B, k_b),
        ),
        R,
    )

    B = torch.where(
        torch.logical_and(R_r == 1, R_m == 1),
        torch.where(
            M == 1,
            R + _cnv_h(B, k_b) - _cnv_h(R, k_b),
            R + _cnv_v(B, k_b) - _cnv_v(R, k_b),
        ),
        B,
    )

    RGB = tstack([R, G, B])

    del R, G, B, k_b, R_r, B_r

    if refining_step:
        RGB = refining_step_Menon2007(RGB, tstack([R_m, G_m, B_m]), M)

    del M, R_m, G_m, B_m

    return RGB



demosaicing_CFA_Bayer_DDFAPD = demosaicing_CFA_Bayer_Menon2007


def refining_step_Menon2007(RGB, RGB_m, M):
   

    R, G, B = tsplit(RGB)
    R_m, G_m, B_m = tsplit(RGB_m)
    M = as_float_array(M)

    del RGB, RGB_m

    # Updating of the green component.
    R_G = R - G
    B_G = B - G

    FIR = torch.ones(3).cuda() / 3

    B_G_m = torch.where(
        B_m == 1,
        torch.where(M == 1, _cnv_h(B_G, FIR), _cnv_v(B_G, FIR)),
        0,
    )
    R_G_m = torch.where(
        R_m == 1,
        torch.where(M == 1, _cnv_h(R_G, FIR), _cnv_v(R_G, FIR)),
        0,
    )

    del B_G, R_G

    G = torch.where(R_m == 1, R - R_G_m, G)
    G = torch.where(B_m == 1, B - B_G_m, G)

    # Updating of the red and blue components in the green locations.
    # Red rows.
    R_r = (torch.any(R_m == 1, axis=1).unsqueeze(0)).permute(1,0) * torch.ones(R.shape).cuda()
    # Red columns.
    R_c = torch.any(R_m == 1, axis=0).unsqueeze(0) * torch.ones(R.shape).cuda()
    # Blue rows.
    B_r = (torch.any(B_m == 1, axis=1).unsqueeze(0)).permute(1,0) * torch.ones(B.shape).cuda()
    # Blue columns.
    B_c = torch.any(B_m == 1, axis=0).unsqueeze(0) * torch.ones(B.shape).cuda()

    R_G = R - G
    B_G = B - G

    k_b = torch.tensor([0.5, 0, 0.5]).cuda()

    R_G_m = torch.where(
        torch.logical_and(G_m == 1, B_r == 1),
        _cnv_v(R_G, k_b),
        R_G_m,
    )
    R = torch.where(torch.logical_and(G_m == 1, B_r == 1), G + R_G_m, R)
    R_G_m = torch.where(
        torch.logical_and(G_m == 1, B_c == 1),
        _cnv_h(R_G, k_b),
        R_G_m,
    )
    R = torch.where(torch.logical_and(G_m == 1, B_c == 1), G + R_G_m, R)

    del B_r, R_G_m, B_c, R_G

    B_G_m = torch.where(
        torch.logical_and(G_m == 1, R_r == 1),
        _cnv_v(B_G, k_b),
        B_G_m,
    )
    B = torch.where(torch.logical_and(G_m == 1, R_r == 1), G + B_G_m, B)
    B_G_m = torch.where(
        torch.logical_and(G_m == 1, R_c == 1),
        _cnv_h(B_G, k_b),
        B_G_m,
    )
    B = torch.where(torch.logical_and(G_m == 1, R_c == 1), G + B_G_m, B)

    del B_G_m, R_r, R_c, G_m, B_G

    # Updating of the red (blue) component in the blue (red) locations.
    R_B = R - B
    R_B_m = torch.where(
        B_m == 1,
        torch.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)),
        0,
    )
    R = torch.where(B_m == 1, B + R_B_m, R)

    R_B_m = torch.where(
        R_m == 1,
        torch.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)),
        0,
    )
    B = torch.where(R_m == 1, R - R_B_m, B)

    del R_B, R_B_m, R_m

    return tstack([R, G, B])
