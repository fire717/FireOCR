import torch
import torch.nn as nn
import torch.nn.functional as F





def conv_block(inp, outp):
    return nn.Sequential(
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, outp ,kernel_size=3,stride=1,padding=1, bias=True),
    )
    


class dense_block(nn.Module):
    def __init__(self, nb_layers, inp, outp, inplace=True):
        super(dense_block, self).__init__()
        self.inplace = inplace

        self.nb_layers = nb_layers
        # self.nb_filter = nb_filter
        self.outp = outp

        convs = []
        for i in range(self.nb_layers):
            convs.append(conv_block(inp+8*i,8))
        self.convs = nn.ModuleList(convs)

        # self.conv1 = conv_block(inp+8*0,8)
        # self.conv2 = conv_block(inp+8*1,8)
        # self.conv3 = conv_block(inp+8*2,8)
        # self.conv4 = conv_block(inp+8*3,8)
        # self.conv5 = conv_block(inp+8*4,8)
        # self.conv6 = conv_block(inp+8*5,8)
        # self.conv7 = conv_block(inp+8*6,8)
        # self.conv8 = conv_block(inp+8*7,8)

    def forward(self, x):
        for conv in self.convs:
            #print(x.shape, self.convs[i])
            cb = conv(x)
            #print(x.shape, cb.shape)
            x = torch.cat((x, cb), 1)
            # self.nb_filter += outp
        # x =  torch.cat((x,self.conv1(x)), 1)
        # x =  torch.cat((x,self.conv2(x)), 1)
        # x =  torch.cat((x,self.conv3(x)), 1)
        # x =  torch.cat((x,self.conv4(x)), 1)
        # x =  torch.cat((x,self.conv5(x)), 1)
        # x =  torch.cat((x,self.conv6(x)), 1)
        # x =  torch.cat((x,self.conv7(x)), 1)
        # x =  torch.cat((x,self.conv8(x)), 1)
        return x



def transition_block(inp, outp):
    return nn.Sequential(
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, outp, kernel_size=1,stride=2,padding=0, bias=False),
        nn.Dropout(0.2)
    )



class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class DenseCNN(nn.Module):
    def __init__(self, input_size=(32,280), n_class=5990, _nb_filter = 64):
        super(DenseCNN, self).__init__()


        self.conv1 = nn.Conv2d(1, _nb_filter,kernel_size=5,stride=2,padding=2, bias=False)
        self.db1 = dense_block(8, _nb_filter,  8)
        self.tb1 = transition_block(128,128)
        self.db2 = dense_block(8, 128, 8)
        self.tb2 = transition_block(192,128)
        self.db3 = dense_block(8, 128, 8)

        self.bn1 = nn.BatchNorm2d(192)
        self.relu1 = nn.ReLU()
        self.dense = torch.nn.Linear(768,n_class)

        
        # self.pool =  nn.AdaptiveMaxPool1d(1)
        # self.task2 = torch.nn.Linear(105,1)

        self._initialize_weights()


    def forward(self, x):
        x = self.conv1(x)#[2, 64, 16, 140]
        # print("1: ", x.shape)
        x = self.db1(x)#[2, 128, 16, 140]
        # print("2: ", x.shape)
        x = self.tb1(x)#[2, 128, 8, 70]
        # print("3: ", x.shape)
        x = self.db2(x)#[2, 192, 8, 70]
        # print("4: ", x.shape)
        x = self.tb2(x)#[2, 128, 4, 35]
        # print("5: ", x.shape)
        x = self.db3(x)#[2, 192, 4, 35]
        # print("6: ", x.shape)
        # print(x.shape)

        x = self.bn1(x)
        # print('1',x)
        x = self.relu1(x)


        x = x.permute(0, 3, 1, 2)#[2, 35, 192, 4]
        # print("7: ", x.shape)

        x = x.view(x.shape[0], x.shape[1], -1)#[2, 35, 768]
        # print("8: ", x.shape)

        x = self.dense(x)#[2, 35, 5990]
        # print("9: ", x.shape)
        return x
        # if mode=='train':
        #     #print(x.shape)#[32, 105, 768]
        #     x = self.pool(x).squeeze()
        #     # print(x.shape)
        #     # b
            
        #     out2 = self.task2(x)
        #     return out1, out2
        # else:
        #     return out1


    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)











if __name__ == '__main__':

    from torchsummary import summary
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model = DenseCNN().cuda()
    print(summary(model, (1, 32, 280)))


    # dummy_input1 = torch.randn(1, 1, 32, 280).cuda()
    # input_names = [ "input1"] #自己命名
    # output_names = [ "output1" ]
    
    # torch.onnx.export(model, dummy_input1, "pose.onnx", 
    #     verbose=True, input_names=input_names, output_names=output_names,
    #     do_constant_folding=True, opset_version=11)

