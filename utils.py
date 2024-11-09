import torch
import os
import torch.nn.functional as F


def phi(x):
    return F.softplus(x)

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def D_uot(x, G, D, D_optimizer, criterion):
    for p in D.parameters():  
        p.requires_grad = True

    real_data = x.cuda()
    real_data.requires_grad = True
        
    D.zero_grad()

    # real D loss
    noise = torch.randn_like(real_data)        
    D_real = D(real_data)
    errD_real = phi(-D_real)
    errD_real = errD_real.mean()
    errD_real.backward(retain_graph=True)
    
    # R1 regularization
    grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=real_data, create_graph=True)[0]
    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
    grad_penalty = 0.2 / 2 * grad_penalty
    grad_penalty.backward()

    # fake D loss
    latent_z = torch.randn(x.shape[0], 100).cuda()
    x_0_predict = G(latent_z)
    D_fake = D(x_0_predict)
    
    errD_fake = phi(D_fake - 0.001 * torch.sum(((x_0_predict-noise).view(noise.size(0), -1))**2, dim=1))
    errD_fake = errD_fake.mean()
    errD_fake.backward()
    errD = errD_real + errD_fake
    D_optimizer.step()

    return errD.data.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()


def G_uot(x, G, D, G_optimizer, criterion):
    for p in D.parameters():
        p.requires_grad = False
            
    real_data = x.cuda()
    real_data.requires_grad = True
            
    G.zero_grad()

    # Generator loss
    noise = torch.randn_like(real_data)
    latent_z = torch.randn(x.shape[0], 100).cuda()
    x_0_predict = G(latent_z)
    D_fake = D(x_0_predict)
    
    err = 0.001 * torch.sum(((x_0_predict-noise).view(noise.size(0), -1))**2, dim=1) - D_fake
    err = err.mean()
    err.backward()
    G_optimizer.step()

    return err.data.item()


def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
