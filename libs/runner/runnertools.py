import os
import time
import torch
import torch.optim as optim



def getSchedu(schedu, optimizer):
    if 'default' in schedu:
        factor = float(schedu.strip().split('-')[1])
        patience = int(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                    mode='max', factor=factor, patience=patience,min_lr=0.000001)
    elif 'step' in schedu:
        step_size = int(schedu.strip().split('-')[1])
        gamma = float(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=-1)
    elif 'SGDR' in schedu: 
        T_0 = int(schedu.strip().split('-')[1])
        T_mult = int(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=T_0, 
                                                            T_mult=T_mult)
    elif 'multi' in schedu:
        milestones = [int(x) for x in schedu.strip().split('-')[1].split(',')]
        gamma = float(schedu.strip().split('-')[2])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1)
    
    else:
        raise Exception("Unkown getSchedu: ", schedu)
    return scheduler

def getOptimizer(optims, model, learning_rate, weight_decay):
    if optims=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optims=='AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optims=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise Exception("Unkown getSchedu: ", optims)
    return optimizer




############### Tools

def clipGradient(optimizer, grad_clip=1):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def writeLogs(cfg, 
            best_epoch, 
            early_stop_value,
            log_time):
    # 可以自定义要保存的字段
    line_list= cfg['log_item']
    log_path = os.path.join(cfg['save_dir'], 'history.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', encoding='utf-8') as f:
            line = ','.join(['timestamps']+line_list+['best_epoch','best_value'])+"\n"
            f.write(line)

    with open(log_path, 'a', encoding='utf-8') as f:

        line_tmp = [log_time] + \
                    ['-'.join([str(v) for v in cfg[x]]) if isinstance(cfg[x],list) else cfg[x] for x in line_list] + \
                    [best_epoch, early_stop_value]
        line = ','.join([str(x) for x in line_tmp])+"\n"
        f.write(line)