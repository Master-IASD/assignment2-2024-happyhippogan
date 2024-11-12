import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision
import copy


from model import Generator, Discriminator
from utils import D_train, G_train, save_models, D_uot, G_uot, EMA




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()

    print("start")
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('samples', exist_ok=True)

    print("end")

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')

    n_samples = 0
    for batch, _ in test_loader:
        for k in range(batch.shape[0]):
            torchvision.utils.save_image(batch[k:k+1], os.path.join('test_data', f'{n_samples}.png'))
            n_samples += 1


    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()


    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 



    # define loss
    criterion = nn.BCELoss() 

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = 1.6e-4, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr = 1e-4, betas=(0.5, 0.999))

    ema = EMA(0.999)
    ema_model = copy.deepcopy(G).cuda()

    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            d_loss = D_uot(x, G, D, D_optimizer, criterion)
            g_loss = G_uot(x, G, D, G_optimizer, criterion)
        print(d_loss)
        print(g_loss)

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

        if epoch == 30:
            with torch.no_grad():
                ema_model = copy.deepcopy(G).cuda()
        if epoch > 30:
            with torch.no_grad():
                ema.update_model_average(ema_model, G)
            if epoch % 10 == 0:
                torch.save(ema_model.state_dict(), os.path.join("checkpoints",'ema.pth'))

        z = torch.randn(64, 100).cuda()
        x = G(z)
        x = x.reshape(64, 28, 28)
        torchvision.utils.save_image(x[1], os.path.join('samples', f'{epoch}.png'))
                
    print('Training done')

        
