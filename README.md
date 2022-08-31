# Pyramid Transformer Net (PTNet)
### [Project](https://github.com/XuzheZ/PTNet) |  [Paper](https://arxiv.org/abs/2105.13993) <br>
Pytorch implementation of PTNet for high-resolution and longitudinal infant MRI synthesis.<br><br>
PTNet: A High-Resolution Infant MRI Synthesizer Based on Transformer  
Xuzhe Zhang<sup>1</sup>, Xinzi He<sup>1</sup>, Jia Guo<sup>2</sup>, Nabil Ettehadi<sup>1</sup>, Natalie Aw<sup>2</sup>, David Semanek<sup>2</sup>, Jonathan Posner<sup>2</sup>, Andrew Laine<sup>1</sup>, Yun Wang<sup>2</sup>  
 <sup>1</sup>Columbia University Department of Biomedical Engineering, <sup>2</sup>CUMC Department of Psychiatry   

## Reminder:

This 2D-only PTNet repo has been deprecated. Please visit our latest [repo](https://github.com/XuzheZ/PTNet3D) containing both 2D and 3D versions with a better data sampling strategy. 

This repo contains the code of our first version preprint paper. This version of PTNet is only designed for pure MAE/MSE loss. **Combining it with adversarial training will significantly impair performance.** If you want to integrate an adversarial training framework, please refer to our **updated version** for the journal paper which introduces substantial improvements (e.g., 3D version, perceptual and adversarial losses).
https://github.com/XuzheZ/PTNet3D

## Usage and Demo

To synthesize high resolution infant brain MRI.

## Prerequisites
- Linux 
- Python3.6
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started
### Installation

git clone https://github.com/XuzheZ/PTNet.git

### Testing
coming soon


### Dataset
In our first version preprint paper, we conducted experiments only on dHCP dataset (http://www.developingconnectome.org/), For more challenging longitudinal tasks, please refer to our updated version for the journal paper: https://github.com/XuzheZ/PTNet3D

### Training
coming soon

## More Training/Test Details
coming soon


## Citation

If you find this useful for your research, please use the following.

```
@article{zhang2021ptnet,
  title={PTNet: A High-Resolution Infant MRI Synthesizer Based on Transformer},
  author={Zhang, Xuzhe and He, Xinzi and Guo, Jia and Ettehadi, Nabil and Aw, Natalie and Semanek, David and Posner, Jonathan and Laine, Andrew and Wang, Yun},
  journal={arXiv preprint arXiv:2105.13993},
  year={2021}
}
```

## Acknowledgments
This code borrows heavily from: [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://github.com/yitu-opensource/T2T-ViT), [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

