import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import random
from typing import Optional
import numpy as np
import torch

from viz import visualize_moe


def set_seed(seed: Optional[int] = 42) -> None:
    """Set all random seeds for reproducibility"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

class EnhancedExpert(nn.Module):
    """Enhanced expert with residual connections and batch norm"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, in_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_dim)
        
        # Skip connection if dimensions change
        self.skip = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim)
        ) if in_dim != hidden_dim else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class EnhancedGate(nn.Module):
    """Enhanced gating network with attention and regularization"""
    def __init__(self, in_dim, num_experts, temperature=0.1):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)*temperature)
        
        # Spatial attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 4, 1),
            nn.BatchNorm2d(in_dim // 4),
            nn.ReLU(),
            nn.Conv2d(in_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Expert selection
        self.expert_gate = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_dim // 2, num_experts)
        )
        
    def forward(self, x, training=True):
        # Spatial attention
        spatial_weights = self.spatial_gate(x)
        x = x * spatial_weights
        
        # Channel attention with both avg and max pooling
        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        combined = torch.cat([avg_feat, max_feat], dim=1)
        
        # Expert selection with temperature scaling
        gates = self.expert_gate(combined)
        if training:
            gates = gates / self.temperature
        
        return F.softmax(gates, dim=1)

class ImprovedMoE(nn.Module):
    """Improved Mixture of Experts with enhanced components and regularization"""
    def __init__(self, num_experts=8, expert_hidden_dim=256, temp=0.1, diversity_coef=0.1):
        super().__init__()
        self.diversity_coef = diversity_coef
        
        # Backbone with larger feature maps
        self.backbone = timm.create_model(
            'efficientnet_b2',
            pretrained=True,
            features_only=True,
            out_indices=(1,)  # Use Stage 1 for 8x8 features
        )
        
        
        # Get feature dimensions
        dummy = torch.randn(2, 3, 32, 32)
        features = self.backbone(dummy)
        self.feature_dim = features[0].shape[1]
        
        # Enhanced gating network
        self.gate = EnhancedGate(self.feature_dim, num_experts, temp)
        
        # Enhanced experts
        self.experts = nn.ModuleList([
            EnhancedExpert(self.feature_dim, expert_hidden_dim)
            for _ in range(num_experts)
        ])
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 10)
        )
    
    def calculate_diversity_loss(self, expert_outputs, method='cosine_abs', eps=1e-8):
        """
        Calculate diversity loss between experts.
        
        Args:
        - expert_outputs: Tensor of shape (batch_size, num_experts, ...)
        - method: String specifying the method to use. Options are:
                'cosine', 'squared_diff', 'kl_div', 'cosine_abs', 'cosine_relu', 'cosine_squared'
        - eps: Small value to avoid division by zero
        
        Returns:
        - diversity_loss: Scalar tensor representing the diversity loss
        """
        batch_size = expert_outputs.size(0)
        num_experts = expert_outputs.size(1)
        expert_outputs_flat = expert_outputs.view(batch_size, num_experts, -1)
        
        if method == 'cosine':
            similarities = torch.matmul(expert_outputs_flat, expert_outputs_flat.transpose(1, 2))
            norms = torch.norm(expert_outputs_flat, dim=2, keepdim=True)
            similarities = similarities / (torch.matmul(norms, norms.transpose(1, 2)) + eps)
            diversity_loss = torch.mean(torch.triu(similarities.mean(0), diagonal=1))
        
        elif method == 'squared_diff':
            differences = (expert_outputs_flat.unsqueeze(2) - expert_outputs_flat.unsqueeze(1)).pow(2).sum(-1)
            diversity_loss = -torch.mean(torch.triu(differences.mean(0), diagonal=1))
        
        elif method == 'kl_div':
            expert_probs = F.softmax(expert_outputs_flat, dim=-1)
            kl_div = F.kl_div(expert_probs.log().unsqueeze(1), expert_probs.unsqueeze(2), reduction='none').sum(-1)
            diversity_loss = -torch.mean(torch.triu(kl_div.mean(0), diagonal=1))
        
        elif method == 'cosine_abs':
            similarities = torch.matmul(expert_outputs_flat, expert_outputs_flat.transpose(1, 2))
            norms = torch.norm(expert_outputs_flat, dim=2, keepdim=True)
            similarities = similarities / (torch.matmul(norms, norms.transpose(1, 2)) + eps)
            diversity_loss = torch.abs(torch.mean(torch.triu(similarities.mean(0), diagonal=1)))
        
        elif method == 'cosine_relu':
            similarities = torch.matmul(expert_outputs_flat, expert_outputs_flat.transpose(1, 2))
            norms = torch.norm(expert_outputs_flat, dim=2, keepdim=True)
            similarities = similarities / (torch.matmul(norms, norms.transpose(1, 2)) + eps)
            diversity_loss = torch.relu(torch.mean(torch.triu(similarities.mean(0), diagonal=1)))
        
        elif method == 'cosine_squared':
            similarities = torch.matmul(expert_outputs_flat, expert_outputs_flat.transpose(1, 2))
            norms = torch.norm(expert_outputs_flat, dim=2, keepdim=True)
            similarities = (similarities / (torch.matmul(norms, norms.transpose(1, 2)) + eps)).pow(2)
            diversity_loss = torch.mean(torch.triu(similarities.mean(0), diagonal=1))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return diversity_loss

    def forward(self, x, return_losses=False):
        features = self.backbone(x)[0]
        
        # Get expert weights
        expert_weights = self.gate(features, self.training)
        
        # Process through experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(features))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Combine expert outputs
        combined = torch.sum(
            expert_outputs * expert_weights.view(-1, expert_weights.size(1), 1, 1, 1),
            dim=1
        )
        
        # Classification
        output = self.classifier(combined)
        
        if return_losses and self.training:
            # Load balancing loss
            expert_usage = expert_weights.mean(0)
            ideal_usage = torch.ones_like(expert_usage) / len(self.experts)
            balance_loss = F.mse_loss(expert_usage, ideal_usage)
            
            diversity_loss = self.calculate_diversity_loss(expert_outputs)
            
            # Auxiliary losses dictionary
            losses = {
                'balance': balance_loss,
                'diversity': diversity_loss * self.diversity_coef
            }
            return output, losses
            
        return output

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, desc='Training'):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, aux_losses = model(inputs, return_losses=True)
        
        # Main classification loss
        task_loss = F.cross_entropy(outputs, targets)
        
        # Combine all losses
        loss = task_loss + aux_losses['balance'] + aux_losses['diversity']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    return 100. * correct / total


def main():
    set_seed(42)
    batch_size = 128
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Enhanced data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.1)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Data loaders
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # Create model
    model = ImprovedMoE(
        num_experts=5,
        expert_hidden_dim=128,
        temp=2.0,
        diversity_coef=0.15
    ).to(device)
    
    # Separate learning rates for different components
    params = [
        {'params': model.backbone.parameters(), 'lr': 1e-4},  # Lower LR for pretrained
        {'params': model.gate.parameters(), 'lr': 3e-4},      # Medium for gate
        {'params': model.experts.parameters(), 'lr': 3e-4},   # Medium for experts
        {'params': model.classifier.parameters(), 'lr': 3e-4} # Medium for classifier
    ]
    
    optimizer = torch.optim.AdamW(params, weight_decay=0.01)
    
    # Calculate exact number of steps
    total_epochs = 100
    total_steps = total_epochs * len(trainloader)  
    
    # OneCycle scheduler with exact steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-4, 2e-4, 2e-4, 2e-4] ,  # Corresponding to param groups
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy='cos',
        final_div_factor=1e4
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(total_epochs):  
        print(f'\nEpoch: {epoch+1}')
        train_loss, train_acc = train_epoch(
            model, trainloader, optimizer, scheduler, device
        )
        
        if epoch % 10 == 0:  
            test_acc = evaluate(model, testloader, device)
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')
            print(f'Test Acc: {test_acc:.3f}%')
            visualize_moe(model, testloader, device,save_to_disk=True,save_path='./plots_dloss')
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Best Test Acc: {best_acc:.3f}%')


if __name__ == '__main__':
    main()