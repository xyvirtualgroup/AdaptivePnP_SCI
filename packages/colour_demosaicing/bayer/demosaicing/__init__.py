# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .bilinear import demosaicing_CFA_Bayer_bilinear
from .malvar2004 import demosaicing_CFA_Bayer_Malvar2004
from .menon2007 import (demosaicing_CFA_Bayer_DDFAPD,
                        demosaicing_CFA_Bayer_Menon2007)
# from .menon2007_tensor import (demosaicing_CFA_Bayer_DDFAPD,
#                         demosaicing_CFA_Bayer_Menon2007)        

# from .Unet import demosaicing_CFA_Bayer_Unet                        

__all__ = []
__all__ += ['demosaicing_CFA_Bayer_bilinear']
__all__ += ['demosaicing_CFA_Bayer_Malvar2004']
# __all__ += ['demosaicing_CFA_Bayer_Unet']
__all__ += ['demosaicing_CFA_Bayer_DDFAPD', 'demosaicing_CFA_Bayer_Menon2007']
