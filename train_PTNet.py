### This code is largely borrowed from pix2pixHD pytorch implementation
### https://github.com/NVIDIA/pix2pixHD

import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
import fractions
from models.models import create_model
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0
import torch.nn as nn
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
ler = opt.lr
mse = torch.nn.MSELoss()
G = create_model(opt)
G.cuda()

optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=0)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter
CE = nn.CrossEntropyLoss()
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
true_label = torch.ones((4,1)).cuda().long()
false_label = torch.zeros((4,1)).cuda().long()
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        # print(data['label'].shape)
        generated = G(Variable(data['label'].cuda()))
        loss_mse = mse(generated,Variable(data['image'].cuda()))
        loss_dict = dict(zip(['MSE'], [loss_mse]))

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        loss = loss_mse
        loss.backward()
        optimizer_G.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ## save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            # model.module.save('latest')
            torch.save(G.state_dict(), os.path.join(opt.checkpoints_dir,opt.name,'latest.pth'))
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        torch.save(G.state_dict(), os.path.join(opt.checkpoints_dir,opt.name, 'ckpt%d%d.pth' % (epoch, total_steps)))
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')


    ## linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        ler -= (opt.lr) / (opt.niter_decay)
        for param_group in optimizer_G.param_groups:
            param_group['lr'] = ler
            print('change lr to ')
            print(param_group['lr'])