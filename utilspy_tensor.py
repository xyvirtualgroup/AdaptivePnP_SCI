''' Utilities '''
import math
import numpy as np
import torch
import os

def mkdir(path):
    
    path=path.strip()        
    path=path.rstrip("\\")    
    
    isExists=os.path.exists(path)   
    
    if not isExists:        
        os.makedirs(path)  
        print (path+' Successfully')
        return True
    else:            
        print (path+' Item existed')
        return False

def worker_init_fn(pid):
    np.random.seed(42+pid)
    torch.manual_seed(42+pid)
    torch.cuda.manual_seed(42+pid)

    
def A_(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI), where multiple coded
    frames are collapsed into a snapshot measurement.
    '''
    return torch.sum(x*Phi, dim=2)  # element-wise product

def At_(y, Phi):
    '''
    Transpose of the forward model. 
    '''
    # (nrow, ncol, nmask) = Phi.shape
    # x = np.zeros((nrow, ncol, nmask))
    # for nt in range(nmask):
    #     x[:,:,nt] = np.multiply(y, Phi[:,:,nt])
    # return x
    return torch.multiply(torch.repeat_interleave(torch.unsqueeze(y,dim=2),Phi.shape[2],axis=2), Phi)

def psnr(ref, img):
    '''
    Peak signal-to-noise ratio (PSNR).
    '''
    mse = torch.mean( (ref - img) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
