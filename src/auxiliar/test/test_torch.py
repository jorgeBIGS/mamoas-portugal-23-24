import torch
import sys
import os
from subprocess import call
print('_____Python, Pytorch, Cuda info____')
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA RUNTIME API VERSION')
#os.system('nvcc --version')
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('_____nvidia-smi GPU details____')
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('_____Device assignments____')
print('Number CUDA Devices:', torch.cuda.device_count())
print ('Current cuda device: ', torch.cuda.current_device(), ' **May not correspond to nvidia-smi ID above, check visibility parameter')
print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))