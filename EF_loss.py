import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def RGB2Gray(imgRGB):
    R = imgRGB[0]
    G = imgRGB[1]
    B = imgRGB[2]
    imgGray = 0.299 * R + 0.587 * G + 0.114 * B
    
    return imgGray


def simpleISP(imgCFA, flag_fiveK, flagCuda=torch.cuda.is_available()):

    bpp = 12
    imgCFA = torch.clamp(imgCFA, min = 0, max = 2 ** bpp - 1)
    if not flag_fiveK:
        CCM = torch.DoubleTensor([[1.685547, -0.501953, -0.183594], [-0.207031, 1.369141, -0.162109], [0.021484, -0.642578, 1.621094]])
        if flagCuda:
            CCM = CCM.cuda()    
        N, H, W = imgCFA.size()
        imgCFA = torch.reshape(imgCFA, (N, H * W))
        imgCFA = torch.matmul(CCM, imgCFA)
        imgCFA = torch.reshape(imgCFA, (N, H, W))
        imgCFA = torch.clamp(imgCFA, min = 0, max = 2**bpp-1)

    imgCFA = imgCFA/(2**bpp - 1)
    img = torch.clamp(imgCFA, min = 1e-12, max = 1)
    imgGamma = torch.pow(img, 1.0/2.2) 
    imgGamma = (torch.clamp(imgGamma, min=0, max=1)).unsqueeze(0) 
    return imgGamma    

def padRepBoard(img, flagCuda = torch.cuda.is_available()):
    N, C, H, W = img.size()
    imgPad = torch.zeros((N, C, H+2, W+2), dtype=torch.double)
    if flagCuda:
        imgPad = imgPad.cuda()
    imgPad[:, :, 1:H+1, 1:W+1] = img
    imgPad[:, :, :, 0] = imgPad[:, :, :, 1]
    imgPad[:, :, :,W+1] = imgPad[:, :, :, W]
    imgPad[:, :, 0,:] = imgPad[:, :, 1,:]
    imgPad[:, :, H+1,:] = imgPad[:, :, H,:]        
    return imgPad

def upsample(x, odd_x, odd_y, kernel):
    img = padRepBoard(x)
    _, c, h, w = img.size()
    R = torch.zeros((c, h*2, w*2), dtype=torch.double)
    if torch.cuda.is_available():
        R = R.cuda()
    R[:,::2,::2] = 4*img
    R = gaussian_conv2d(R.unsqueeze(0), kernel)
    output = R[:, :, 2 : h*2-odd_x-2, 2 : w*2-odd_y-2]

    return output

def downsample(x, kernel):
    #return x[:, :, ::2, ::2]    
    R = gaussian_conv2d(x, kernel)    
    R = R[:,:, ::2, ::2]
    return R

def EFweights(images, w_c, w_s, w_e):
    """
        images is a list of n images with size (3,W,H) 
        return the weights for each imgs
    """
    n_frame = len(images)
    r = images[0].size(2)
    c = images[0].size(3)
    weights_sum = torch.zeros((r, c), dtype=torch.double)
    weights = torch.zeros((n_frame, r,c),  dtype=torch.double)
    if torch.cuda.is_available():
        weights_sum = weights_sum.cuda()
        weights = weights.cuda()
    for i in range(n_frame):
        img = images[i][0]
        filter = torch.DoubleTensor([[0,1,0], [1,-4,1], [0,1,0]]).unsqueeze(0).unsqueeze(0)
        W = torch.ones((r, c))        
        if torch.cuda.is_available():
            filter = filter.cuda()
            W = W.cuda()
        # contrast
        img_gray = RGB2Gray(img)
        laplacian = F.conv2d(img_gray.unsqueeze(0).unsqueeze(0), filter, padding = 1)[0][0]
        W_contrast = torch.pow(torch.abs(laplacian), w_c)
        # print("W_contrast", torch.min(W_contrast), torch.max(W_contrast))
        W = W * W_contrast
    
        # saturation
        std = torch.sqrt(torch.sum( (img - torch.mean(img, dim=0)) **2, dim = 0)/3 + 1e-12)
        W_saturation = torch.pow(std, w_s)
        # print("W_saturation", torch.min(W_saturation), torch.max(W_saturation))
        W = W * W_saturation
    
        # well-exposedness
        sigma2 = 0.2
        W_exposedness = torch.pow(torch.prod(torch.exp(-torch.pow(img - 0.5, 2)/(2*sigma2**2)), dim = 0), w_e)
        W = W * W_exposedness + 1e-12
        # print("W_exposedness", torch.min(W_exposedness),torch.max(W_exposedness))
        weights_sum += W
        weights[i,:,:] = W
    # normalization
    for i in range(n_frame):
        weights[i] /= weights_sum

    return weights

def gaussian_kernel(channels=3):
    # Create Gaussian Kernel. In Numpy
    # ax = np.linspace(-(size - 1)/ 2., (size-1)/2., size)
    # xx, yy = np.meshgrid(ax, ax)
    # kernel = np.exp(-0.5 * (np.square(xx)+ np.square(yy)) / np.square(sigma))
    # kernel /= np.sum(kernel)
    # # Change kernel to PyTorch. reshapes to (channels, 1, size, size)
    # kernel_tensor = torch.DoubleTensor(kernel)
    kernel = np.asmatrix(np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])).T
    kernel = kernel.dot(kernel.T)
    kernel_tensor = torch.DoubleTensor(kernel)
    kernel_tensor = kernel_tensor.repeat(channels, 1, 1, 1)
    return kernel_tensor

def gaussian_conv2d(x, g_kernel):
    #Assumes input of x is of shape: (minibatch, depth, height, width)
    #Infer depth automatically based on the shape
    channels = g_kernel.size(0)
    padding = g_kernel.size(-1) // 2 # Kernel size needs to be odd number
    if len(x.size()) != 4:
        raise IndexError('Expected input tensor to be of shape: (batch, depth, height, width) but got: ' + str(x.shape))
    y = F.conv2d(x, weight=g_kernel, stride=1, padding=padding, groups=channels)
    return y

def create_laplacian_pyramid(x, kernel, levels):
    """return pyramid list"""
    # upsample = torch.nn.Upsample(scale_factor=2) # Default mode is nearest: [[1 2],[3 4]] -> [[1 1 2 2],[3 3 4 4]]
    pyramids = []
    current_x = x
    for level in range(1,levels):
        gauss_filtered_x = gaussian_conv2d(current_x, kernel)
        down = downsample(gauss_filtered_x, kernel)
        odd_x = 2* down.size(2) - gauss_filtered_x.size(2)
        odd_y = 2* down.size(3) - gauss_filtered_x.size(3)
        up = upsample(down, odd_x, odd_y, kernel)
        laplacian = current_x - up
        pyramids.append(laplacian)
        current_x = down 
    pyramids.append(current_x)
    return pyramids

def get_gaussian_pyramid_weights(weights, kernel, levels):
    """Return the Gaussian Pyramid of the Weight map of all images, weight size = [N,1,W,H]"""
    weight_pyramids = []
    for index in range(weights.size(0)): #img index
        current_x = weights[index].unsqueeze(0).unsqueeze(0)
        pyramids=[]
        pyramids.append(current_x)
        for i in range(1, levels):
            gauss_filtered_x = gaussian_conv2d(current_x, kernel)
            down = downsample(gauss_filtered_x, kernel)
            pyramids.append(down)
            current_x = down 
        weight_pyramids.append(pyramids)
    return weight_pyramids

def reconstruct_image(weight, img_pyramid, kernel):
    """weight a list with len = num_img,  weigth[i] list of pyramid for ith image, same as img_pyramid
       output img with size = [n, 3, h ,w]
    """ 
    #tmpT = transforms.ToPILImage()
    num_img = len(img_pyramid)
    height_pyr = len(img_pyramid[0])        
    result_pyramid = [] 
    for lev in range(height_pyr):
        temp = torch.zeros_like(img_pyramid[0][lev],  dtype=torch.double)
        for i in range(num_img):
            temp += weight[i][lev] * img_pyramid[i][lev]
        result_pyramid.append(temp)

    result_image = result_pyramid[-1]
    for lev in range(height_pyr - 2, -1, -1):
        odd_x = 2* result_image.size(2) - result_pyramid[lev].size(2)
        odd_y = 2* result_image.size(3) - result_pyramid[lev].size(3)
        up = upsample(result_image, odd_x, odd_y, kernel)
        result_image =  result_pyramid[lev] + up
        result_image[result_image < 0] = 0
        result_image[result_image > 1] = 1 

    return result_image

class EF_loss(nn.Module):
    def __init__(self, n_frame = 3):
        super(EF_loss, self).__init__()
        self.n_frame = n_frame
        self.bpp = 12
        self.w_s = 1.0
        self.w_c = 1.0
        self.w_e = 1.0
        self.kernel = gaussian_kernel()
        self.weight_kernel = gaussian_kernel(channels=1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
            self.weight_kernel = self.weight_kernel.cuda()
            
    def fusion(self, img, weights=None):
        """
            img is a list of imgs with size (1,3,H,W)
        """
        level = np.floor(np.log2(np.min((img[0].size(2), img[0].size(3))))).astype(np.int64)
        if weights == None:
            weights = EFweights(img, self.w_c, self.w_s, self.w_e)
        weight_pyramid = get_gaussian_pyramid_weights(weights, self.weight_kernel, level)
        img_laplacian = []
        for k in range(self.n_frame):
            img_laplacian.append(create_laplacian_pyramid(img[k], self.kernel, level))
        img_result = reconstruct_image(weight_pyramid, img_laplacian, self.kernel)
        # img_result = torch.round(img_result*(2**8-1))/(2**8-1)

        return img_result, weights
            

    def forward(self, imgCFA, flag_fiveK, scale, gt, mask = None, mask_weight = 20):
        """
            gt size = (n,3,H,W) where n is the batch size
        """
        img_result = []
        loss = 0
        mse = 0
        weights = []
        for i in range(gt.size(0)):
            img_simpleISP=[]
            for j in range(self.n_frame):
                img = simpleISP(imgCFA[i] * scale[i,j], flag_fiveK[i])   
                img_simpleISP.append(torch.stack([img[0,0,:,:], img[0,1,:,:], img[0,2,:,:]]).unsqueeze(0))	
            fuseImg, w = self.fusion(img_simpleISP)
            img_result.append(fuseImg)
            weights.append(w)
            error = (img_result[-1] - gt[i])**2
            mse += torch.mean(error)
            if mask == None:
                loss += torch.mean(error)
            else:
                loss += torch.mean(error + error * mask[i] * mask_weight)
        img_result = torch.cat(img_result, dim=0)
        weights = torch.cat(weights, dim=0)

        loss /= gt.size(0)
        mse /= gt.size(0)
        psnr = 10 * torch.log10(1. / (mse+1e-12))

        return loss, psnr, img_result, weights

 