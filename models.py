import torch

class Net(torch.nn.Module):
    def __init__(self, activation=''):
        super(Net, self).__init__()
        
        self.d1 = torch.nn.Linear(1, 32)
        self.d2 = torch.nn.Linear(32, 32)
        self.d3 = torch.nn.Linear(32, 32)
        self.d4 = torch.nn.Linear(32, 1)
        
        
        self.tanh = torch.nn.Tanh()
        self.ReLU = torch.nn.ReLU()
        self.LReLU = torch.nn.LeakyReLU(0.01)
        self.c = torch.nn.Parameter(torch.tensor([0.1]))
        self.c.requires_grad = True

        self.activation = activation
        
    def func(self, x):
        return  torch.log(torch.exp(x)+1)
    def ReLuPlus(self,x):
        return torch.maximum(self.ReLU(self.c),x)
    def forward(self, phi):
        
        x = self.tanh(self.d1(phi))
        x = self.tanh(self.d2(x))
        x = self.tanh(self.d3(x))
        
        if self.activation == 'ReLU':
            u = self.ReLU(self.d4(x))
        elif self.activation == 'LReLU':
            u = self.LReLU(self.d4(x))
        elif self.activation == 'func':
            u = self.func(self.d4(x))
        elif self.activation == 'ReLUPlus':
            u = self.ReLuPlus(self.d4(x))
            print('------------------>',self.c)
        else:
            u = self.d4(x)
            
        return u
    
    
class FCN(torch.nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = torch.nn.Tanh
        self.ReLU = torch.nn.ReLU()
        self.fcs = torch.nn.Sequential(*[
                        torch.nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = torch.nn.Sequential(*[
                        torch.nn.Sequential(*[
                            torch.nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = torch.nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x