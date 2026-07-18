import torch
import torchvision
import numpy
import scipy
import sklearn
import stable_baselines3
import gymnasium
import matplotlib
import seaborn
import pandas
import tqdm
import opacus
import hdbscan

print('='*40)
print('All imports successful')
print('='*40)
print(f'PyTorch      : {torch.__version__}')
print(f'Torchvision  : {torchvision.__version__}')
print(f'Numpy        : {numpy.__version__}')
print(f'SB3          : {stable_baselines3.__version__}')
print(f'Gymnasium    : {gymnasium.__version__}')
print(f'Opacus       : {opacus.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')