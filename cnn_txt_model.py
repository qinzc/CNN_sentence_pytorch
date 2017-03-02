import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.functional as TF
import torch.optim as optim
import pdb

from torch.autograd import Variable

class CNN_Txt_Net(nn.Module):
    def __init__(self, llen, word_len = 300, filter_shp = [3,4,5], num_filter =100, hidden_units=[100,2], activation = 'relu'):
        super(CNN_Txt_Net, self).__init__()

        # Now, define the layers.
        # We need to know the pool_size
        conv_layers = []
        self.filter_shp = filter_shp
        self.num_filter = num_filter
        self.llen = llen
        for shp in filter_shp:
            conv_layer = nn.Conv2d( 1, num_filter, kernel_size = (shp, word_len))
            conv_layers.append(conv_layer)
            
        self.conv_list = nn.ModuleList(conv_layers)

        hunits = [ len(filter_shp) * num_filter]
        hunits.extend(hidden_units)

        fcs = []
        for h_in, h_out in zip(hunits, hunits[1:]):
            fcs.append(nn.Linear(h_in, h_out))

        self.fcs = nn.ModuleList(fcs)

        self.act = activation

    def forward(self, x):
        outputs = []
        for i in xrange(len(self.conv_list)):
            x_p = F.relu(F.max_pool2d(self.conv_list[i](x), (self.llen - self.filter_shp[i] + 1, 1)))
            outputs.append(x_p)
            
        opt = TF.stack(outputs, dim = -2) # shoule be batch x numfilter x 1
        #opt = torch.cat(outputs, dim = -2)
        opt = opt.view(-1, self.num_filter * len(self.conv_list))

        for i in xrange(len(self.fcs)):
            #opt = nn.Dropout(0.5)(opt)
            opt = F.dropout(opt, p = 0.2)
            opt = self.fcs[i](opt)
            if self.act == 'relu':
                opt = F.relu(opt)
            else:
                opt = F.relu(opt)
        
        #return nn.LogSoftmax()(opt)
        return F.log_softmax(opt)
