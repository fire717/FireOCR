
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




class OCRLoss(nn.Module):
    def __init__(self, blank_id=0, device='cpu'):
        super().__init__()

        self.device = device
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean')


    def l1(self, pre,target):
        return torch.mean(torch.abs(pre - target))
        
    def forward(self, x, targets, input_lengths, target_lengths):

        # print(x[0].shape, x[1].shape)
        # l1
        # print(target_lengths)#[4, 105, 22]) tensor([24, 30, 28, 30]
        # pre_lens = decodeOutput(x)
        if len(x)==2:
            loss1 = self.l1(x[1], target_lengths)
            # print(loss1)
            # b

            # ctc
            log_probs = F.log_softmax(x[0], -1)
            # print(log_probs.shape)
            log_probs = log_probs.transpose(1,0) # N,T,C -> T,N,C
            targets = targets.view(-1)
            loss2 = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)#/log_probs.shape[1]
            


            return loss1, loss2
        else:
            log_probs = F.log_softmax(x, -1)
            # print(log_probs.shape)
            log_probs = log_probs.transpose(1,0) # N,T,C -> T,N,C
            targets = targets.view(-1)
            loss2 = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)#/log_probs.shape[1]

            return loss2





if __name__ == '__main__':



    device = torch.device("cpu")

    #x = torch.randn(2,2)
    x = torch.tensor([[0.1,0.7,0.2]])
    y = torch.tensor([1])
    print(x)

    loss_func = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_func(x,y)
    print("loss1: ",loss)

    # loss_func = Focalloss().to(device)
    # loss = loss_func(x,y)
    # print("loss2: ",loss)
    

    weight_loss = torch.DoubleTensor([1,1,1]).to(device)
    loss_func = FocalLoss(gamma=0, weight=weight_loss).to(device)
    loss = loss_func(x,y)
    print("loss3: ",loss)
    

    # weight_loss = torch.DoubleTensor([2,1]).to(device)
    # loss_func = Focalloss(gamma=0.2, weight=weight_loss).to(device)
    # loss = loss_func(x,y)
    # print("loss4: ",loss)