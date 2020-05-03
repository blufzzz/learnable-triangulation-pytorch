# reference: https://github.com/zhangboshen/A2J
import cv2
import torch
import numpy as np
import scipy.io as scio
import os
from PIL import Image
import model as model
from tqdm import tqdm
import torch.utils.data
from torch.utils.data import Dataset

def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x

def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x
    
def dataPreprocess(index, img, keypointsUVD, center, mean, std, lefttop_pixel, rightbottom_pixel, validIndex, xy_thres=95, depth_thres=150):
 
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 
 
    new_Xmin = max(lefttop_pixel[index,0,0], 0)
    new_Ymin = max(rightbottom_pixel[index,0,1], 0)  
    new_Xmax = min(rightbottom_pixel[index,0,0], img.shape[1] - 1)
    new_Ymax = min(lefttop_pixel[index,0,1], img.shape[0] - 1)

    
    imCrop = img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize,dtype = 'float32')  # H*W*C

    imgResize[np.where(imgResize >= center[index][0][2] + depth_thres)] = center[index][0][2]
    imgResize[np.where(imgResize <= center[index][0][2] - depth_thres)] = center[index][0][2] 
    imgResize = (imgResize - center[index][0][2])

    imgResize = (imgResize - mean) / std

    label_xy = np.ones((keypointsNumber, 2), dtype = 'float32') 
    
    label_xy[:,0] = (keypointsUVD[validIndex[index],:,0].copy() - new_Xmin)*cropWidth/(new_Xmax - new_Xmin) # x
    label_xy[:,1] = (keypointsUVD[validIndex[index],:,1].copy() - new_Ymin)*cropHeight/(new_Ymax - new_Ymin) # y
    
    
    imageOutputs[:,:,0] = imgResize

    labelOutputs[:,1] = label_xy[:,0]
    labelOutputs[:,0] = label_xy[:,1] 
    
    labelOutputs[:,2] = (keypointsUVD[validIndex[index],:,2] - center[index][0][2])   # Z  
    
    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label


class ICVL(Dataset):

    def __init__(self, 
                trainingImageDir=???, 
                center, 
                lefttop_pixel, 
                rightbottom_pixel, 
                keypointsUVD, 
                validIndex):

        self.trainingImageDir = trainingImageDir
        self.mean = MEAN
        self.std = STD
        self.center = center
        self.lefttop_pixel = lefttop_pixel
        self.rightbottom_pixel = rightbottom_pixel
        self.keypointsUVD = keypointsUVD
        self.validIndex = validIndex
        self.xy_thres = xy_thres
        self.depth_thres = depth_thres

        fx = 240.99
        fy = 240.96
        u0 = 160
        v0 = 120
                                # trainingImageDir
                                #validIndex_train = np.load('../data/icvl/validIndex.npy')
                                #TrainImgFrames = len(validIndex_train)

        # validIndex
        TestImgFrames = 1596
        validIndex_test = np.arange(TestImgFrames)

        keypointsNumber = 16
        cropWidth = 176
        cropHeight = 176
        batch_size = 8
        xy_thres = 95
        depth_thres = 150

        data_root = '/media/hpc2_storage/ibulygin/learnable-triangulation-pytorch/A2J/data/'
        keypointsfile = os.path.join(data_root,'icvl/icvl_keypointsUVD_test.mat')
        center_file = os.path.join(data_root,'icvl/icvl_center_test.mat')
        MEAN = np.load(os.path.join(data_root,'icvl/icvl_mean.npy'))
        STD = np.load(os.path.join(data_root,'icvl/icvl_std.npy'))

        # keypointsUVD
        keypointsUVD_test = scio.loadmat(keypointsfile)['keypoints3D'].astype(np.float32)   
        
        # center
        center_test = scio.loadmat(center_file)['centre_pixel'].astype(np.float32)
        centre_test_world = pixel2world(center_test.copy(), fx, fy, u0, v0)
        
        centerlefttop_test = centre_test_world.copy()
        centerlefttop_test[:,0,0] = centerlefttop_test[:,0,0]-xy_thres
        centerlefttop_test[:,0,1] = centerlefttop_test[:,0,1]+xy_thres
        centerrightbottom_test = centre_test_world.copy()
        centerrightbottom_test[:,0,0] = centerrightbottom_test[:,0,0]+xy_thres
        centerrightbottom_test[:,0,1] = centerrightbottom_test[:,0,1]-xy_thres

        # lefttop_pixel
        test_lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
        # rightbottom_pixel
        test_rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)

    def __getitem__(self, index):

        depth = scio.loadmat(self.trainingImageDir + str(self.validIndex[index]+1) + '.mat')['img']       
         
        data, label = dataPreprocess(index, depth, self.keypointsUVD, self.center, self.mean, self.std, \
            self.lefttop_pixel, self.rightbottom_pixel, self.validIndex, self.xy_thres, self.depth_thres)
       

        return data, label
    
    def __len__(self):
        return len(self.center)

