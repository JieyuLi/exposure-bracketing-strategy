import torch
import torch.nn as nn
import math

class HistogramSpacialNet(nn.Module):

    def __init__(self, num_classes=3, bins_num=128):
        super(HistogramSpacialNet, self).__init__()
        
        self.bins_num = bins_num

        self.features = nn.Sequential( 
            nn.Conv1d(61, 64, kernel_size=4, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=4, padding=0), 
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=4, padding=0), 
            nn.ReLU(inplace=True) 
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2*256, 256),
        )
        self._initialize_weights()

    def forward(self, x):
        f = self.features(x)
        x = f.view(f.size(0), 2*256) 
        x = self.classifier(x)
        return f, x

    def _initialize_weights(self):
        cnt = 0
        for m in self.modules():
            
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                cnt += 1
                if cnt == 3:
                    m.weight.data.normal_(0, 0.05)
                else:
                    m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

def histogramspacialnet(pretrained = False, bins_num = 128):
    model = HistogramSpacialNet(bins_num=bins_num)
    return model

class l2norm(nn.Module):
    def __init__(self):
        super(l2norm,self).__init__()

    def forward(self,input,epsilon = 1e-7):
        assert len(input.size()) == 2,"Input dimension requires 2,but get {}".format(len(input.size()))
        
        norm = torch.norm(input,p = 2,dim = 1,keepdim = True)
        output = torch.div(input,norm+epsilon)
        return output


class scaleTime(nn.Module):
    def __init__(self):
        super(scaleTime,self).__init__()
        self.Tmax = torch.DoubleTensor([2**8])#11
        if torch.cuda.is_available():
            self.Tmax = self.Tmax.cuda() 

    def forward(self, input):
        input[:,0] += 0.5
        input[:,-1] -= 1.0
        input = torch.sigmoid(input)
        output = torch.exp(2 * (input - 0.5) * torch.log(self.Tmax))
        return output

class ExposureStrategyNet(nn.Module):
    def __init__(self, pretrained = False, num_frame = 3, bins_num = 128):
        super(ExposureStrategyNet, self).__init__()
        self.histogram = histogramspacialnet(pretrained, bins_num).double() #output size 1024
        self.l2norm = l2norm()
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_frame),
        ).double()  
        self.scaleTime = scaleTime() 
        self._initialize_weights()

    def forward(self, hist):
        his_conv, fhis = self.histogram(hist)
        output = self.classifier(fhis)
        output = self.scaleTime(output)        
        output, _ = torch.sort(output, descending=True)
        return his_conv, fhis, output

    def _initialize_weights(self):
        cnt = 0
        for m in self.classifier.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                cnt += 1
                if cnt == 2:
                    m.weight.data.normal_(0, 0.05)
                else:
                    m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()