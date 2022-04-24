import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
import cv2
import rawpy




def simpleCFA(imgRaw):
    """
        convert bayer image to rgb image (just do a simple binning)
        :param imgRaw: a bayer image with format BGGR HxW
        :return imgCFA: an rgb image with size H/2xW/2x3
    """
    H, W = imgRaw.shape
    imgRawPad = np.zeros((H+2, W+2))
    imgRawPad[1:H+1, 1:W+1] = imgRaw
    imgRawPad[:,0] = imgRawPad[:,2]
    imgRawPad[:,W+1] = imgRawPad[:,W-1]
    imgRawPad[0,:] = imgRawPad[2,:]
    imgRawPad[H+1,:] = imgRawPad[H-1,:]
    imgCFA = np.zeros((H//2, W//2, 3))
    # G channel
    imgCFA[:,:,1] = (imgRawPad[2:H+1:2, 1:W+1:2] + imgRawPad[1:H+1:2, 2:W+1:2])/2.0
    # R channel
    kernel = np.array([[1/16, 3/16], [3/16, 9/16]], dtype=np.float64)
    imgCFA[:,:,0] = cv2.filter2D(imgRawPad[:H:2, :W:2], -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    # B channel
    kernel = np.array([[9/16, 3/16], [3/16, 1/16]], dtype=np.float64)
    imgCFA[:,:,2] = cv2.filter2D(imgRawPad[1:H+1:2, 1:W+1:2], -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)

    return imgCFA

class DatasetHDRNetRLCephName(data.Dataset):
    def __init__(self, data_dir, data_list, frames_num =10, height = None, width = None, bins_num=256):
        super(DatasetHDRNetRLCephName, self).__init__()

        self.data_dir = data_dir
        self.frames_num = frames_num
        self.height = height
        self.width= width

        self.bins_num = bins_num

        self.names = []
        # self.labels = []

        self.bpp = 12
        self.CCM = np.array([[1.685547, -0.501953, -0.183594], [-0.207031, 1.369141, -0.162109], [0.021484, -0.642578, 1.621094]]).astype('float64')

        fin = open(data_list)
        lines = fin.readlines()

        for line in lines:
            line = line.strip().split('\n')
            self.names.append(line[0])
            # self.labels.append(int(line[1]))

        fin.close()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        """
            imgRaw with original size
            imgCFA [3 X width X height]: imgRaw -> demosaicing -> resize
            ldr [3 X width X height]: imgCFA -> clip -> color correction -> gamma 
            hist: 1 X 256
            target: 3 X width X height

        """
        TT = transforms.ToTensor()
        if os.path.exists(os.path.join(self.data_dir, self.names[index],'img_hdr.npy')):
            hdr_path = os.path.join(self.data_dir, self.names[index], 'img_hdr.npy')
            imgRaw = np.load(hdr_path).astype(np.float64) #relative
            imgCFA = simpleCFA(imgRaw)
            flag_fiveK = False
        elif os.path.exists(os.path.join(self.data_dir, self.names[index],'img_hdr.dng')):
            hdr_path = os.path.join(self.data_dir, self.names[index], 'img_hdr.dng')
            img_dng = rawpy.imread(hdr_path)
            imgprocess = img_dng.postprocess(no_auto_bright=True, gamma=(1,1), use_camera_wb=True, output_bps=16) #, 
            imgCFA = imgprocess.astype(np.float64)
            flag_fiveK = True
        else:
            print(hdr_path, 'not found')
            os._exit(1)
            
        if self.height != None and self.width != None:
            imgCFA = cv2.resize(imgCFA, (self.width, self.height), interpolation=cv2.INTER_LINEAR).astype(np.float64)
        
        meta = np.loadtxt(os.path.join(self.data_dir, self.names[index], 'iso.txt')).astype(np.float64)        
        EV0 = meta[0] * meta[1] * 1e-9        
        exposeTime = torch.DoubleTensor([meta[1]*1e-9])
        
        phi = np.loadtxt(os.path.join(self.data_dir, self.names[index], 'phi.txt')).astype(np.float64)        
        phiMin = torch.DoubleTensor([phi[0]])
        phiMax = torch.DoubleTensor([phi[1]])

        target = Image.open(os.path.join(self.data_dir, self.names[index], 'ex2.png'))
        target = np.asarray(target).astype(np.float64)
        target = target/255.0
        if self.height != None and self.width != None:
            target = cv2.resize(target, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        target = TT(target)

        

        ldr = Image.open(os.path.join(self.data_dir, self.names[index], 'normal.png'))
        ldr = np.asarray(ldr).astype(np.float64) #0-255
        if self.height != None and self.width != None:
            ldr = cv2.resize(ldr, (self.width, self.height), interpolation=cv2.INTER_LINEAR).astype(np.float64)

        hist_list = []
        imgLuma = 0.2126 * ldr[:,:,0] + 0.7152 * ldr[:,:,1] + 0.0722 * ldr[:,:,2] 
        a_MAX = 2**8-1
        H, W = imgLuma.shape
        hist = torch.DoubleTensor(np.histogram(imgLuma.flatten(), self.bins_num, [0, a_MAX])[0] / (H * W))
        hist_list.append(hist)        # 3*3
        h = int(H/3)
        w = int(W/3)
        for i in range(3):
            for j in range(3):
                hist =  torch.DoubleTensor(np.histogram(imgLuma[i*h:(i+1)*h, j*w:(j+1)*w].flatten(), self.bins_num, [0, a_MAX])[0] /(h*w))
                hist_list.append(hist)
        # 7*7
        h = int(H/7)
        w = int(W/7)
        for i in range(7):
            for j in range(7):
                hist =  torch.DoubleTensor(np.histogram(imgLuma[i*h:(i+1)*h, j*w:(j+1)*w].flatten(), self.bins_num, [0, a_MAX])[0] /(h*w))
                hist_list.append(hist)
        
        
        hist_list.append(torch.ones_like(hist) * (torch.log2(phiMin) - torch.DoubleTensor([4.06]))/torch.DoubleTensor([5.95]))
        hist_list.append(torch.ones_like(hist) * (torch.log2(phiMax) - torch.DoubleTensor([10.48]))/torch.DoubleTensor([5.67]))
        #hist_list.append(torch.ones_like(hist) * torch.log2(phiMin))
        #hist_list.append(torch.ones_like(hist) * torch.log2(phiMax))
        
        hist = torch.stack(hist_list, dim = 0)     
        
        ldr = TT(ldr)
        imgCFA = TT(imgCFA)
        EV0 = torch.DoubleTensor([EV0])
        
        return target, phiMin, phiMax, imgCFA, ldr, hist, EV0, exposeTime, flag_fiveK, self.names[index]