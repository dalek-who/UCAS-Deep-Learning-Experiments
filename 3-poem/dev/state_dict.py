from torch import nn
from torch import optim
from torch.optim import lr_scheduler

model = nn.RNN(3, 5)
optimizer = optim.Adam(model.parameters())
lr_lambda = lambda epoch: 1 / (1 + 0.5 * epoch)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

print(model.state_dict())
print(optimizer.state_dict())
print(scheduler.state_dict())