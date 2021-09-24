### This code is largely borrowed from pix2pixHD pytorch implementation
### https://github.com/NVIDIA/pix2pixHD

import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import nibabel as nib
import numpy as np
import torch
import time

opt = TestOptions().parse(save=False)
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
G = create_model(opt)
G.cuda()
G.eval()
fld = os.path.join(opt.checkpoints_dir, opt.name)

ckpt_lst = [i for i in os.listdir(fld) if i.endswith('.pth') and not os.path.isdir(os.path.join(fld, i))]
print(ckpt_lst)


for ckpts in ckpt_lst:
    if not os.path.isdir(os.path.join(fld,'Synthesized_2D_' + ckpts)):
        os.mkdir(os.path.join(fld,'Synthesized_2D_' + ckpts))
        os.mkdir(os.path.join(fld,'Synthesized_3D_' + ckpts))

    G.load_state_dict(torch.load(os.path.join(fld,ckpts)))

    start = time.time()
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        generated = G(data['label'].cuda())

        img_path = data['path']

        label=data['label'].cpu().float().numpy()
        gen_img=generated.data[0].cpu().float().numpy()
        gen_img = nib.Nifti1Image(gen_img[0], np.eye(4))
        nib.save(gen_img, os.path.join(fld,'Synthesized_2D_' + ckpts,img_path[0].split('/')[-1]) )
    end = time.time()
    print('exec time:')
    print(end-start)

webpage.save()
