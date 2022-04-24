import torch
import torch.nn as nn

class NoiseModel(nn.Module):

    def __init__(self, exposureTime, flag = torch.cuda.is_available()):
        super(NoiseModel, self).__init__()
        self.params = {                
            'G1':{ 'S':0.1940969078, 'R0': 0.07927728, 'R1': 2.79670275},
            'R': { 'S':0.1877266707, 'R0': 0.05800109, 'R1': 1.68789693},
            'B': { 'S':0.1348817769, 'R0': 0.07155223, 'R1': 2.4669988},
            'G2': {'S':0.206432242, 'R0': 0.08001507, 'R1': 2.98408152}
        }
        self.d = torch.DoubleTensor([(4095-256)/4095])
        self.c = torch.DoubleTensor([0])
        self.maxAg = torch.DoubleTensor([12800])
        self.time = exposureTime
        #self.EV0 = EV0
        self.I_max = torch.DoubleTensor([2**12 - 1])
        if flag:
            self.d = self.d.cuda()
            self.c = self.c.cuda()
            #self.EV0 = self.EV0.cuda()
            self.maxAg = self.maxAg.cuda()
            self.I_max = self.I_max.cuda()
            self.time = self.time.cuda()
            
    
    def genNoiseImg_CFA(self, I, iso):
        """
        generate a single noisy img given iso and rgb image
        """
        noise_var = torch.zeros_like(I)
        if torch.cuda.is_available():
            noise_var = noise_var.cuda()
        ag = torch.min(self.maxAg, iso)
        dg = iso / ag
        x = I * self.d / dg
        if len(x.size()) == 4:
            x = x[0]
        A,B = self.get_A_B_iso(ag, 'G', 0, 1)
        noise_var[1] = A * x[1] + B 
        A,B = self.get_A_B_iso(ag, 'R', 0, 1)
        noise_var[0] = A * x[0] + B
        A,B = self.get_A_B_iso(ag, 'B', 0 ,1)
        noise_var[2] = A * x[2] + B
        noise_var = noise_var / (self.d / dg)**2       
        noise =  torch.sqrt(noise_var) * torch.randn_like(x)
        output = I + noise
        
        return noise_var, output


    def get_A_B_iso(self, iso, color, c, d, flag=torch.cuda.is_available()):
        """
        given specific iso setting, return the parameter A and B for the noise variance estimation ~= A * x + B
        :param iso: ISO setting
        :param color: specify which color channel ['R' | 'G' | 'B']
        :param c, d: the noise calibration might be conducted in original raw space (without normalization / just do the darklevel substrction).
                    If your signal has the conversion: X' = (X - c) / d, noise estimation should be converted correspondingly
        :return: A and B list for 4 color channel
        """
        if color.upper() in 'RB':
            p = self.params[color.upper()]
        elif color.upper() == 'G':
            p = self.params['G1']  # Sometimes calibration on G2 is not accurate, so only use G1's data
        else:
            raise Exception('You can only choose from R/G/B channel to get noise model')
        S = torch.DoubleTensor([p['S']])
        R0 = torch.DoubleTensor([p['R0']])
        R1 = torch.DoubleTensor([p['R1']])
        if flag:
            S = S.cuda()
            R0 = R0.cuda()
            R1 = R1.cuda()
        A =  S * (iso / 100)
        B = R0 * (iso / 100) ** 2 + R1
        A_ = A / d
        B_ = (B + A * c) / (d ** 2)
        return A_, B_
    
    def Noise(self, phi, iso, t):
        p = self.params['G1']            
        S = torch.DoubleTensor([p['S']])
        R0 = torch.DoubleTensor([p['R0']])
        R1 = torch.DoubleTensor([p['R1']])
        if torch.cuda.is_available():
            S = S.cuda()
            R0 = R0.cuda()
            R1 = R1.cuda()

        iso_d = iso / (100 * self.d)
        I = phi * iso * t
        dg = torch.max(iso/self.maxAg, torch.DoubleTensor([1]).cuda())/self.d
        noise_var = S * I *iso_d  + R0 * iso_d ** 2 + R1 * dg**2   
        return noise_var
    
        
    def mergedSNR(self, phi, scale_abs, t):

        iso = scale_abs/t
        I = phi * scale_abs
        snr = torch.DoubleTensor([0])
        if torch.cuda.is_available():
            snr = snr.cuda()
        for i in range(scale_abs.size(0)):
            noise = self.Noise(phi, iso[i], t)
            snr += (I[i]**2)/noise
        return snr
        

    def forward(self, phiMin, phiMax, EV0, scale, weight_noise, weight_range, if_print= False):
        """
            EV0: N * 3 * 1
            scale: N * 3 * 1 relative to EV0
            phiMin/phiMax: N * 1
            
        """
        loss = 0
        for j in range(scale.size(0)):
            scale_abs = scale[j] * EV0[j]
                       
            cost_list = []
            range_cost = []
           
            phis =  self.I_max/scale_abs
            if if_print:
                print("phi:", phis)
                print("phiRange:", phiMin, phiMax)
            phis_in = phis[(phis>phiMin[j]) & (phis<phiMax[j])]
            scale_in = scale_abs[phis>=phiMin[j]]
            
            for i in range(len(phis_in)):
                if i + 1 < len(scale_in):
                    snr = torch.log2(self.mergedSNR(phis_in[i], scale_in[i+1:], self.time[j]))
                    cost_list.append(weight_noise* snr)
                    if if_print:
                        print(phis[i].item(), "snr", (10*torch.log10(2**(snr))).item(), "cost", -cost_list[-1].item())
            scale_min = scale_abs[phis>phiMin[j]]                             
            if len(scale_min) != 0:
                snr = torch.log2(self.mergedSNR(phiMin[j], scale_min, self.time[j]))
                cost_list.append(weight_noise* snr)
                if if_print:
                    print("phiMin", phiMin[j].item(), "snr", (10*torch.log10(2**(snr))).item(), "cost", -cost_list[-1].item())
                    
            phi_min = phis[phis<=phiMin[j]]                             
            for i in range(len(phi_min)):
                temp = weight_range *(torch.exp(torch.log2(phiMin[j]) - torch.log2(phi_min[i])) - 1)
                range_cost.append(temp)
                if if_print:
                    print("!!phiMin", phiMin[j].item(), "phi", phi_min[i], "cost", temp)


            phi_max = phis[phis>phiMax[j]]                             
            if len(phi_max) == 0:
                temp = weight_range * (torch.exp(torch.log2(phiMax[j]) - torch.log2(self.I_max/scale_abs[-1])) - 1)
                range_cost.append(temp) 
                if if_print:
                    print("!!phiMax", phiMax[j].item(), "phi2", self.I_max/scale_abs[-1], "cost", temp)

            if len(range_cost) != 0:
                loss += torch.sum(torch.stack(range_cost))
            if len(cost_list) != 0:
                loss -= torch.min(torch.stack(cost_list))
        loss /= scale.size(0)
                
        return loss#, imgRawNoise
