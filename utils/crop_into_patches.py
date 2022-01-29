from utils import tools
import cv2 as cv
import numpy as np
import torch
def patches(origin, patch_size, stride, batch_size):
    # origin 8 3 132 220
    size = batch_size*(origin.shape[2]//patch_size)*(origin.shape[3]//patch_size)

    o = []
    data = torch.from_numpy(np.zeros((size,3, patch_size, patch_size) , dtype='uint8')) # 522939
    cnt = 0
    for i in range(batch_size):# n images

        origin = (origin.to(torch.uint8))
        [_, _, hei, wid] = origin.shape
        # for count in range(6): # 6 patches per img
        for x in range(0, hei - patch_size + 1,stride):
            for y in range(0, wid - patch_size + 1,stride):
                subim_origin = origin[i,:,x: x + patch_size, y: y + patch_size]
                data[cnt,:,:,:] = subim_origin
                cnt+=1


    # pic = data.numpy()
    # for i in range(pic.shape[0]):
    #     name = str(i) + '.jpg'
    #     cv.imwrite(name, np.array([pic[i, 0, :, :], pic[i, 1, :, :], pic[i, 2, :, :]]).T)

    return data


