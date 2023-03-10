import sys
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime

from data_utils import *


def data_loader(names, normalization_type='No', call_path='..'):
    
    x_stars, y_stars, r_stars, u_stars, phi_stars,stars_info = Stars_Info(names,call_path,show=False)
    
    
    e_stars = {name: stars_info.loc[name]['e'] for name in names}
    p_stars = {name: stars_info.loc[name]['p [Au]'] for name in names}
    
    
    if normalization_type == 'No':
        x_stars, y_stars = x_stars, y_stars
        e_stars, p_stars = e_stars, p_stars
        
#         min_u = np.asarray([np.min(u_stars[name]) for name in u_stars])
#         stars_info['min u'] = min_u
        display(stars_info)
        return x_stars, y_stars, r_stars, u_stars, phi_stars, e_stars, p_stars
    
    elif isinstance(normalization_type,dict): 
        if list(normalization_type.keys())[0] == 'EllipseArea':
            coeff = list(normalization_type.values())[0]
            tmp   = {name: EllipseNorm(x_stars[name],y_stars[name],e_stars[name],p_stars[name],coeff) for name in names}
            x_new = {name: tmp[name][0] for name in names}
            y_new = {name: tmp[name][1] for name in names}
            phi_new = phi_stars
            r_new = {name: np.sqrt(x_new[name]**2 + y_new[name]**2) for name in names}
            u_new = {name: 1/r_new[name] for name in names}

            e_new = {name: tmp[name][2] for name in names}
            p_new = {name: tmp[name][3] for name in names}
            
#             min_u = np.asarray([np.min(u_new[name]) for name in u_new])
#             stars_info['min u (Elliptic normalization)']=min_u
            display(stars_info)
        return x_new, y_new, r_new, u_new, phi_new, e_new, p_new
    
def train_convert(*datas):
    d = [torch.from_numpy(list(data.values())[0]).float().view(-1,1) for data in datas]
    return d

def phi_transform(phi):
    phi = list(phi.values())[0]
    
    idx = np.where(np.abs(np.diff(phi, prepend=phi[0])) > np.pi)[0]
    for i in idx:
        phi[i:] += 2 * np.pi * np.sign(phi[i - 1] - phi[i])
    phi -= np.min(phi)
    return phi

def trainer(net, Phys_part, phi_train, u_train, phi_phys, e, p, name, coeff=1, call_path='.', **steps):
    
    path = call_path+f"/model_weights/"
    number = str(0)
    if not os.path.exists(path+name+'_'+number):
        os.makedirs(path+name+'_0')
    else:
        files = [i for i in os.listdir(path) if name+'_' in i]
        number=str(int(files[-1].split('_')[-1]) + 1)
        os.makedirs(path+name+f'_{number}')
    
    batch_size, epochs, lr = steps['batch_size'], steps['epochs'], steps['lr']
    phi_phys = phi_phys.requires_grad_(True)
    
    net.train()
    Phys_part.train()
    optimizer = torch.optim.Adam(list(net.parameters()) + list(Phys_part.parameters()), lr=lr)
    for epoch in range(1, epochs + 1):
        trainloader = torch.utils.data.DataLoader(torch.arange(len(phi_phys)), 
                                                  batch_size=batch_size, 
                                                  shuffle=False, 
                                                  drop_last=False)
        running_loss = 0
        
        for i, batch_indxs in enumerate(trainloader, 0):
            optimizer.zero_grad()
            
            u_data_target = u_train #[batch_indxs,:]
            phi_data_target = phi_train #[batch_indxs,:]
            phi_phys_target = phi_phys[batch_indxs,:]

            u_data_pred = net(phi_data_target)
            u_phys_pred = net(phi_phys_target)
            
            Loss_data = torch.nn.MSELoss()(u_data_pred,u_data_target)
            
            Loss_phys = Phys_part(u_phys_pred, phi_phys_target)
            
#             Loss_phys = Phys_part(phi_phys_target)
            Loss_phys = Loss_phys #(1e-4)*Loss_phys
            Loss = Loss_data + Loss_phys
            running_loss += Loss.item()
            Loss.backward()
            optimizer.step()
            percent = (i + 1) / len(trainloader) * 100
            
            print(f'Epoch {epoch} \t{Loss.item():.8f}\t{Loss_data.item():.8f}\t{Loss_phys.item():.8f}\t{Phys_part.p.item():.3f}\t{percent:.0f}%', end='\r')
        
#             sys.stdout.flush()
        
        print('\nAverage loss =', running_loss / len(trainloader))
        
        if epoch % 500 == 0:
            torch.save(net.state_dict(), path + f'{name}' + '_' + f'{number}' + f'/model_{name}.pt')
            torch.save(Phys_part.state_dict(), path + f'{name}' + '_' + f'{number}' + f'/Phys_part_{name}.pt')
            
            net.eval()
            u = net(phi_phys)
            
            e_ell = list(e.values())[0]
            p_ell = list(p.values())[0]
            phi_ell = np.linspace(0,4*np.pi,200) 
            x_ell,y_ell,r_ell = ellipse(phi_ell,e_ell,p_ell)
            
            p_ell_pred = Phys_part.p.item()
            e_ell_pred = np.sqrt(1-(np.pi*coeff*p_ell_pred**2)**(2/3))
            x_ell_pred,y_ell_pred,r_ell_pred = ellipse(phi_ell,e_ell_pred,p_ell_pred)
            
            fig = plt.figure(figsize=(20,5))
            fig.subplots_adjust(wspace=0)

            gs = gridspec.GridSpec(1, 2, width_ratios=[2,1])
    
            ax1 = plt.subplot(gs[0], projection='polar')
            ax2 = plt.subplot(gs[1])
            
            ax1.scatter(phi_phys.detach().numpy(), (1/u).detach().numpy(), marker='+', label='Prediction')
            ax1.scatter(phi_train.detach().numpy(), (1/u_train).detach().numpy(), marker='+', label='Train data')
            ax1.plot(phi_ell, r_ell_pred, label=f'Pred. ellipse, e={e_ell_pred:.3f}, p={p_ell_pred:.3f}')
            ax1.plot(phi_ell, r_ell, label=f'Real ellipse, e={e_ell:.3f}, p={p_ell:.3f}' )
            
            ax2.plot(phi_phys.detach().numpy(), u.detach().numpy())
            ax2.scatter(phi_phys.detach().numpy(), u.detach().numpy(), marker='+')
            ax2.scatter(phi_train.detach().numpy(),u_train.detach().numpy())
            ax2.plot(phi_ell, 1/r_ell)
            
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.8))

            plt.show()
            plt.close()
            
            if epoch % 5000 == 0:
                print('-------Save Figure-------')
                fig.savefig(path + f'{name}' + '_' + f'{number}' + f'/plot_{name}_{epoch}.png',dpi=300)
                