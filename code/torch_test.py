import os
os.environ['TORCH_COMPILE'] = '0'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['TORCHINDUCTOR_DISABLE'] = '1'
os.environ['TORCHFX_TRACER'] = '0'

import torch
print("Torch version:", torch.__version__)
print("Trying to create a simple model and optimizer...")
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print("Success!")