import torch

print ('Torch version', torch.__version__)
print ('Torch cuda version', torch.version.cuda)
print ('Cuda', torch.cuda.is_available())
print ('CUDNN', torch.backends.cudnn.enabled)
print("Number of available GPUs: {}".format(torch.cuda.device_count()))
