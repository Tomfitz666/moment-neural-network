
import torch
import torch.nn as nn
import numpy as np
from mnn.mnn_core.mnn_utils import Mnn_Core_Func
from momentactivationtrio import MnnActivateTrio
from RMNN_model import RMNN

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torchvision.transforms import ToTensor
import time




def input_encoder(data, scale=1.0):
   
    
    data = torch.flatten(data, start_dim=1)        # (B, 784)
    input_mean = data * scale
    
    input_var = torch.abs(input_mean) + 1e-6        # 方差 >= 0
    input_cov = torch.diag_embed(input_var)          # (B, 784, 784)
    return input_mean, input_cov





class MNISTRecurrentMNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10, 
                 num_steps=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
       
        self.rmnn = RMNN(N=hidden_dim, M=input_dim)
        
        self.readout = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, images):
        """
        Args:
            images: (B, 1, 28, 28)
        Returns:
            logits: (B, num_classes)
        """
        B = images.size(0)
        device = images.device
        
        
        # mu_ff: (B, 784),  C_ff: (B, 784, 784)
        mu_ff, C_ff = input_encoder(images, scale=1.0)
        
        
        # mu: (B, hidden_dim),  C: (B, hidden_dim, hidden_dim)
        mu = torch.zeros(B, self.hidden_dim, device=device)
        C = torch.eye(self.hidden_dim, device=device).unsqueeze(0).expand(
            B, -1, -1) * 0.1
        
        
        for t in range(self.num_steps):
            mu, C = self.rmnn(mu, C, mu_ff, C_ff)
        
       
        logits = self.readout(mu)
        
        return logits
    
def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        logits = model(data)
        loss = criterion(logits, target)
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
        
        if batch_idx % 200 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch} [{batch_idx * len(data):>5d}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.4f}  "
                  f"Acc: {100. * correct / total:.1f}%  "
                  f"Time: {elapsed:.1f}s")
    
    avg_loss = total_loss / total
    avg_acc = 100. * correct / total
    return avg_loss, avg_acc


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            logits = model(data)
            loss = criterion(logits, target)
            
            total_loss += loss.item() * data.size(0)
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    
    avg_loss = total_loss / total
    avg_acc = 100. * correct / total
    return avg_loss, avg_acc



if __name__ == "__main__":
    
    batch_size = 32
    hidden_dim = 64
    num_steps = 3          # BPTT 展开步数
    num_epochs = 5
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Hidden dim: {hidden_dim}, BPTT steps: {num_steps}")
    
    
    train_dataset = datasets.MNIST(
        './datasets/', train=True, download=True,
        transform=transforms.Compose([
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  
        ])
    )
    test_dataset = datasets.MNIST(
        './datasets/', train=False, download=True,
        transform=transforms.Compose([
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    
    model = MNISTRecurrentMNN(
        input_dim=784,
        hidden_dim=hidden_dim,
        num_classes=10,
        num_steps=num_steps
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
   
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
   
    best_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}  LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        
        scheduler.step()
        
        print(f"\n  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f}  Test  Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_rmnn_mnist.pth')
            print(f"  ✅ New best! Saved model.")
        
        
        model.eval()
        with torch.no_grad():
            sample_data = next(iter(test_loader))[0][:8].to(device)
            mu_ff, C_ff  = input_encoder(sample_data)
            
            
            mu = torch.zeros(8, hidden_dim, device=device)
            C = torch.eye(hidden_dim, device=device).unsqueeze(0).expand(8, -1, -1).clone() * 0.1
            
            for t in range(num_steps):
                mu, C = model.rmnn(mu, C, mu_ff, C_ff)
                active = (mu > 1e-6).float().mean().item()
                mu_norm = mu.norm(dim=1).mean().item()
                print(f"    Step {t}: active={active:.1%}, |mu|={mu_norm:.4f}")

        model.train()
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best test accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")