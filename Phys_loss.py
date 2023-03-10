import torch
from models import Net

class Phys_Loss_first_order(torch.nn.Module):
    def __init__(self):
        super(Phys_Loss_first_order, self).__init__()
        self.e = torch.nn.Parameter(torch.tensor([1.]))  #torch.nn.Parameter(torch.rand(1))
        self.e.requires_grad = True
        self.register_parameter('e', self.e)
        
        self.mu = torch.nn.Parameter(torch.tensor([1.]))
        self.mu.requires_grad = True
        self.register_parameter('mu', self.mu)
        
        self.M = torch.nn.Parameter(torch.tensor([1.]))
        self.M.requires_grad = True
        self.register_parameter('M', self.M)
        
        self.Lambda = torch.nn.Parameter(torch.tensor([1.]))
        self.Lambda.requires_grad = True
        self.register_parameter('Lambda', self.Lambda)

        
    def forward(self, u_phys_pred, phi_phys):
        
        du_dphi = torch.autograd.grad(u_phys_pred, phi_phys, torch.ones_like(u_phys_pred), create_graph=True)[0]
        physics = du_dphi**2 - ((self.mu/self.M)**2) * (self.e**2 - 1) - 2*(self.mu/self.M)*u_phys_pred\
                  + u_phys_pred**2 - (self.Lambda*self.mu)/((3*self.M**2)*(u_phys_pred**2) +1e-5)*1e-23
        
        return torch.mean(physics**2)
    
    
    
class Phys_Loss_second_order(torch.nn.Module):
    def __init__(self):
        super(Phys_Loss_second_order, self).__init__()
        
        self.mu = torch.nn.Parameter(torch.tensor([10.]))
        self.mu.requires_grad = True
        self.register_parameter('mu', self.mu)
        
        self.M = torch.nn.Parameter(torch.tensor([500.]))
        self.M.requires_grad = True
        self.register_parameter('M', self.M)
        
        self.Lambda = torch.nn.Parameter(torch.tensor([1.]))
        self.Lambda.requires_grad = True
        self.register_parameter('Lambda', self.Lambda)
        
        self.net = Net()

        
    def forward(self, phi_phys):
        self.net.train()
        
        u_phys_pred = self.net(phi_phys)
        
        du_dphi = torch.autograd.grad(u_phys_pred, phi_phys, torch.ones_like(u_phys_pred), create_graph=True)[0]
        ddu_dphi = torch.autograd.grad(du_dphi, phi_phys, torch.ones_like(du_dphi), create_graph=True)[0]
        
        physics = ddu_dphi - self.mu/self.M + u_phys_pred + self.Lambda*self.mu/(3*(u_phys_pred ** 3)+1e-5)*1e-23
        
        return torch.mean(physics**2)
    

    
class Phys_Loss_old(torch.nn.Module):
    def __init__(self):
        super(Phys_Loss_old, self).__init__()
        
        self.p = torch.nn.Parameter(torch.tensor([1.]))
        self.p.requires_grad = True
        self.register_parameter('p', self.p)
        
        self.A = 1#(6*torch.pi)**2
        
        
    def forward(self,u_phys_pred, phi_phys):
        
        
        du_dphi  = torch.autograd.grad(u_phys_pred, phi_phys, torch.ones_like(u_phys_pred), create_graph=True)[0]
        ddu2_dphi = torch.autograd.grad(du_dphi, phi_phys, torch.ones_like(du_dphi), create_graph=True)[0]
        
        physics = ddu2_dphi+self.A*(u_phys_pred-1/self.p)
        
        return torch.mean(physics**2)
    
    
class Phys_Loss_new(torch.nn.Module):
    def __init__(self, initial_condition=True):
        super(Phys_Loss_new, self).__init__()
        
        self.p = torch.nn.Parameter(torch.tensor([1.]))
        self.p.requires_grad = True
        self.register_parameter('p', self.p)
        
        self.e = torch.sqrt(1-(torch.pi*self.p**2)**(2/3))
        
        self.u_0 = (1+self.e)/self.p
        
        self.init = initial_condition
        
    def forward(self,u_phys_pred,phi_phys):
        
        
        du_dphi  = torch.autograd.grad(u_phys_pred, phi_phys, torch.ones_like(u_phys_pred), create_graph=True)[0]
        ddu2_dphi = torch.autograd.grad(du_dphi, phi_phys, torch.ones_like(du_dphi), create_graph=True)[0]
        
        physics = ddu2_dphi+(u_phys_pred-1/self.p)
        ode_loss = torch.mean(physics**2) 
        
        if self.init:
            du_dphi_0 = du_dphi[0]
            condition = du_dphi_0**2
            Loss = ode_loss + condition 
            
            return Loss
        else:
            return ode_loss
    