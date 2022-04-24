import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
import dataset.dataset as dataset
import models.decision as decision
import loss.EF_loss as EFloss
import loss.noise_loss as noise_model
import time

def arg_parse():
    parser = argparse.ArgumentParser(description='exposure bracketing strategy' )
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('ckpt', metavar='DIR', help='path to checkpoints')
    parser.add_argument('--test-list', metavar='DIR', help='path to test list')
    parser.add_argument('--results', type=str, metavar='DIR',help='results')
    parser.add_argument('--score-path', type=str, metavar='DIR', help='save the decision score in a new list')
  
    args = parser.parse_args()
    return args

def print_args(args):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    for arg,content in args.__dict__.items():
        print("{}:{}".format(arg,content))
    print("\n")

def test():
    global args, A
    args = arg_parse()
    print_args(args)

    
    policy = decision.ExposureStrategyNet(pretrained=False)


    criterion = nn.MSELoss().cuda()

    print( '===> loading weights ...')
    policy_weights = args.ckpt
    if policy_weights:
        if os.path.isfile(policy_weights):
            print("=====> loading checkpoint '{}".format(policy_weights))
            checkpoint = torch.load(policy_weights, map_location=torch.device('cpu'))
            policy.load_state_dict(checkpoint['state_dict'])
            print('epoch', checkpoint['epoch'])
        else:
            print("=====> no checkpoint found at '{}'".format(policy_weights))

    policy.eval()

    tmpT = transforms.ToPILImage()

    if not os.path.exists(args.results):
        os.makedirs(args.results)
    output_path = args.results

    test_set = dataset.DatasetHDRNetRLCephName(args.data, args.test_list, 3, 
                                               height = None, width = None, bins_num=128)
    test_loader = DataLoader(dataset=test_set, num_workers = 0, batch_size = 1, shuffle=False)
    
    fout = open(os.path.join(output_path, 'score.txt'), 'w')

    cnt = 0
    psnr_sum = 0.
    loss_sum = 0.
    noise_snr_sum = 0.
    overall_snr_sum = 0.
    time_sum = 0.
    total_params = sum(p.numel() for p in policy.parameters())
    print("model size: ",total_params)

    num_img = 0
    EFcriterion = EFloss.EF_loss()
    policy.cuda()

    for i, (gt, phiMin, phiMax, imgCFA, ldr, hist, EV0, exposeTime, flag_fiveK, name) in enumerate(test_loader):
        noise = noise_model.NoiseModel(exposeTime)
        print(i)
        hist = hist.cuda()
        imgCFA = imgCFA.cuda()
        ldr = ldr.cuda()
        gt = gt.cuda()
        phiMin = phiMin.cuda()
        phiMax = phiMax.cuda()
        exposeTime = exposeTime.cuda()
        EV0 = EV0.cuda()
        with torch.no_grad():
            start_time = time.time()
            _, _, scale = policy(hist)
            end_time = time.time()
            time_sum += end_time - start_time
            num_img += 1
            print(name[0])
            scale, _ = torch.sort(scale, descending=True)
            iso = scale*EV0/noise.time
            
            loss, psnr, fusion_output, weights = EFcriterion(imgCFA, flag_fiveK, scale, gt)
            fusion_output= torch.clamp(fusion_output, min=0.0, max=1.0)
 
            mse = criterion(fusion_output[0], gt[0])
            psnr = 10 * torch.log10(torch.max(gt[0])**2/ mse)
            psnr = psnr.item()
            psnr_sum += psnr

            img_noise = []
            for j in range(3):            
                _, imgNoise = noise.genNoiseImg_CFA(imgCFA[0]*scale[0][j], iso[0][j])
                imgNoise = EFloss.simpleISP(imgNoise, flag_fiveK) 
                img_noise.append(imgNoise)

            fusionNoise, _ = EFcriterion.fusion(img_noise, weights)
         
            fusionNoise= torch.clamp(fusionNoise, min=0.0, max=1.0)
            mse = torch.mean((fusion_output - fusionNoise)**2)
            noise_snr = 10 * torch.log10((torch.mean(fusion_output[0]))**2/mse)   
            noise_snr_sum += noise_snr

            mse = criterion(fusionNoise, gt)
            overall_snr = 10 * torch.log10(1.0/mse)
            overall_snr_sum += overall_snr

            output_noise = tmpT(torch.clamp(fusionNoise[0, :, :, :], 0, 1).to(torch.float).data.cpu())
            output_noise.save(os.path.join(output_path, name[0] + '_noise.png'))

            print("EV: 2**", torch.log2(scale).cpu().numpy())
            print("ISO:", iso.cpu().numpy())

            fout.write(name[0] + ' ' + str(torch.log2(scale).cpu().numpy()) + ' ' + str(iso.cpu().numpy()) + ' '  + str(psnr) +   ' ' + str(noise_snr.item()) + ' ' + str(overall_snr.item()) +'\n')
            print("psnr:", psnr)
            print("noise_snr", noise_snr.item())
            print("loss: ", loss.item())

            cnt += 1
    psnr_avg = psnr_sum / float(num_img)
    print('psnr: ', psnr_avg)


    noise_snr_avg = noise_snr_sum / float(num_img)
    print('noise_snr: ', noise_snr_avg)
    

    loss_avg = loss_sum / float(num_img)
    print('loss: ', loss_avg)
    
    overall_snr_avg = overall_snr_sum / float(num_img)
    print('overall_snr: ', overall_snr_avg)

    fout.write("pnsr:" + str(psnr_avg) + '\n')
    fout.write("snr:" + str(noise_snr_avg.item()) +  '\n')
    fout.write("overall_snr:" + str(overall_snr_avg.item()) +  '\n')
    fout.write("time:" + str(time_sum/float(num_img)) +  '\n')
    fout.close()

if __name__ == '__main__':
    test()