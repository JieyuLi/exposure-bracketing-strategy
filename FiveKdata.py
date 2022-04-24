import shutil
import numpy as np
import torchvision.transforms as transforms
import rawpy
import os
import loss.EF_loss as EFloss
import torch
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='preprocessing fiveK dataset' )
    parser.add_argument('--data-path', metavar='DIR', help='path to dataset')
    parser.add_argument('--data-list', metavar='DIR', help='path to data list')
    parser.add_argument('--output-path',  metavar='DIR', help='path to dataset')
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    data_dir = args.data_path
    data_list = args.data_list
    output_path = args.output_path
    
    tmpT = transforms.ToPILImage()
    TT = transforms.ToTensor()
    fin = open(data_list)
    lines = fin.readlines()
    names = []
    meta = dict()
    for i in range(len(lines)):
        line = lines[i]
        line = line.strip().split(' ')
        names.append(line[0])
        meta[names[-1]] = [line[1], line[2]]
    fin.close()
    
    
    scale_0 = torch.FloatTensor([[2, 1, 2**-1, 2**-2, 2**-3, 2**-4]])
    tone = EFloss.EF_loss(len(scale_0[0]))

    for j in range(2):
    
        if meta[names[j]][0] =='1/0':   
            continue
    
      
        path_dir = os.path.join(output_path, names[j])
        if not os.path.exists(path_dir):          
            os.makedirs(path_dir)
        
        print(names[j])
        
        shutil.copyfile(os.path.join(data_dir,  names[j]+'.dng'), os.path.join(path_dir, 'img_hdr.dng'))
        img = rawpy.imread(os.path.join(data_dir,  names[j]+'.dng'))
        imgprocess = img.postprocess(no_auto_bright=True, gamma=(1,1), use_camera_wb=True, output_bps=16)
        imgCFA = imgprocess

        ldr = img.postprocess(use_camera_wb=True, output_bps=8)
        ldr = np.round(np.clip((imgCFA/(2**16-1))**(1/2.2)*(2**8-1), a_min=0, a_max= 2**8 -1)).astype(np.float64)
        output = tmpT(TT(ldr/(2**8-1)).to(torch.float32))
        output.save(os.path.join(path_dir, 'normal.png'))

        EV0 = eval(meta[names[j]][0]) * float(meta[names[j]][1])        
        phi_ordered = np.sort(imgCFA[imgCFA>=1],axis=None) 
        H = len(phi_ordered)
        phiMin = phi_ordered[int(0.02*H)]/EV0
        phiMax = phi_ordered[int(0.98*H)]/EV0
        
        fout = open(os.path.join(path_dir,'phi.txt'), 'w')
        fout.write(str(phiMin)+'\n')
        fout.write(str(phiMax))
        fout.close()
        
        fout = open(os.path.join(path_dir,'iso.txt'), 'w')
        fout.write(meta[names[j]][1]+'\n')
        fout.write(str(float(eval(meta[names[j]][0]))*1e9))
        fout.close()
    
        scale = scale_0
        imgCFA = TT(imgCFA.astype(np.int32)).double().unsqueeze(0)
        lossEF, psnr, fusionClean, weights = tone(imgCFA.cuda(), [True], scale.cuda(), torch.zeros_like(imgCFA).cuda(), torch.zeros_like(imgCFA).cuda())

        output = tmpT(torch.clamp(fusionClean[0], 0, 1).to(torch.float32).data.cpu())
        output.save(os.path.join(path_dir, 'ex2.png'))


if __name__=='__main__':
    main()