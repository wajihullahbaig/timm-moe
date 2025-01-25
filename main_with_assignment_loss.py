from collections import defaultdict
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
from typing import Dict, Optional
import numpy as np
import torch

from viz import visualize_moe, visualize_moe_expert_map

def create_expert_assignments(num_classes: int, num_experts: int) -> Dict[int, list]:
    """Create balanced label assignments for experts"""
    assignments = {}
    labels_per_expert = num_classes // num_experts
    remaining = num_classes % num_experts

    start_idx = 0
    for expert_idx in range(num_experts):
        num_labels = labels_per_expert + (1 if expert_idx < remaining else 0)
        assignments[expert_idx] = list(range(start_idx, start_idx + num_labels))
        start_idx += num_labels

    return assignments

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
    def __init__(self, in_dim, num_experts, temperature=0.1, expert_assignments=None):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)*temperature)
        self.expert_assignments = expert_assignments
        
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
            nn.Dropout(0.3),
            nn.Linear(in_dim // 2, num_experts)
        )

    def compute_soft_masks(self, labels):
        """Compute soft assignment masks based on expert assignments"""
        batch_size = labels.size(0)
        num_experts = len(self.expert_assignments)
        
        soft_masks = torch.zeros(batch_size, num_experts, device=labels.device)
        for expert_idx, assigned_labels in self.expert_assignments.items():
            mask = torch.isin(labels, torch.tensor(assigned_labels, device=labels.device))
            soft_masks[:, expert_idx] = mask.float()
        
        # Normalize masks
        row_sums = soft_masks.sum(dim=1, keepdim=True)
        soft_masks = soft_masks / (row_sums + 1e-8)
        return soft_masks

    def forward(self, x, labels=None):
        # Spatial attention
        spatial_weights = self.spatial_gate(x)
        x = x * spatial_weights
        
        # Channel attention with both avg and max pooling
        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        combined = torch.cat([avg_feat, max_feat], dim=1)
        
        logits = self.expert_gate(combined)
        
        if self.training and labels is not None:
            # Add small noise during training
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
            
            # Get soft assignment masks
            soft_masks = self.compute_soft_masks(labels)
            
            # Combine network output with soft guidance
            routing_weights = F.softmax(logits / self.temperature, dim=1)
            mixing_factor = torch.sigmoid(self.temperature)
            final_weights = mixing_factor * routing_weights + (1 - mixing_factor) * soft_masks
            
            return final_weights
        
        return F.softmax(logits / self.temperature, dim=1)

class ImprovedMoE(nn.Module):
    def __init__(self, num_experts=8, expert_hidden_dim=256, temp=0.1):
        super().__init__()

        
        # Create expert assignments
        self.expert_assignments = create_expert_assignments(10, num_experts)  # 10 for CIFAR10
        
        # Backbone
        self.backbone = timm.create_model(
            'efficientnet_b2',
            pretrained=True,
            features_only=True,
            out_indices=(1,)
        )
        
        # Get feature dimensions
        dummy = torch.randn(2, 3, 32, 32)
        features = self.backbone(dummy)
        self.feature_dim = features[0].shape[1]
        
        # Enhanced gating network with assignments
        self.gate = EnhancedGate(self.feature_dim, num_experts, temp, 
                                self.expert_assignments)
        
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
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, 10)
        )


    def calculate_routing_loss(self, expert_weights, labels):
        """New routing loss based on expert assignments"""
        batch_size = labels.size(0)
        num_experts = len(self.expert_assignments)
        
        # Create target assignment matrix
        target_assignments = torch.zeros(batch_size, num_experts, 
                                      device=labels.device)
        for expert_idx, assigned_labels in self.expert_assignments.items():
            mask = torch.isin(labels, torch.tensor(assigned_labels, 
                                                 device=labels.device))
            target_assignments[:, expert_idx] = mask.float()
        
        # Normalize target assignments
        target_assignments = target_assignments / (target_assignments.sum(dim=1, 
                                                keepdim=True) + 1e-5)
        # Clamp to ensure numerical stability
        expert_weights = torch.clamp(expert_weights, min=1e-5)
        # Compute KL divergence between expert weights and target assignments
        routing_loss = F.kl_div(expert_weights.log(), target_assignments, 
                               reduction='batchmean')
        
        return routing_loss
    
    def calculate_diversity_loss(self, expert_outputs, method='kl_div', eps=1e-8):
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


    def calculate_balance_loss(self, expert_weights, labels, num_classes):
        """Calculate balance loss based on class distribution"""
        batch_size = labels.size(0)
        num_experts = expert_weights.size(1)
        
        # Get class distribution in current batch
        class_counts = torch.bincount(labels, minlength=num_classes)
        class_distribution = class_counts.float() / batch_size
        
        # Map class distribution to expert count
        ideal_usage = F.adaptive_avg_pool1d(
            class_distribution.view(1, 1, -1),
            num_experts
        ).squeeze()
        
        # Get actual expert usage
        expert_usage = expert_weights.mean(0)
        balance_loss = F.mse_loss(expert_usage, ideal_usage)
        
        return balance_loss

    def forward(self, x, labels=None, n_classes=None, return_losses=False):
        # Get backbone features
        features = self.backbone(x)[0]
        
        # Get expert weights with labels for training
        expert_weights = self.gate(features, labels if self.training else None)
        invalid_tensors = check_invalid_values(expert_weights)
            
        if invalid_tensors:
            raise ValueError(f"Invalid values (NaN or Inf) found in the following tensors: {', '.join(invalid_tensors)}")
        
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
            balance_loss = self.calculate_balance_loss(expert_weights,labels,n_classes)
            diversity_loss = self.calculate_diversity_loss(expert_weights)
            routing_loss = self.calculate_routing_loss(expert_weights, labels)
            invalid_tensors = check_invalid_values(balance_loss, diversity_loss, routing_loss)
            
            if invalid_tensors:
                raise ValueError(f"Invalid values (NaN or Inf) found in the following tensors: {', '.join(invalid_tensors)}")
            
            losses = {
                'balance_loss': balance_loss,
                'diversity_loss':diversity_loss,
                'routing_loss': routing_loss
            }
            return output, losses
            
        return output


def check_invalid_values(*tensors):
    invalid_tensors = []
    for i, tensor in enumerate(tensors):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            invalid_tensors.append(f"Tensor {i}")
    return invalid_tensors

def train_epoch(model, loader, optimizer, scheduler, n_classes, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    loss_components = defaultdict(float)
    
    for inputs, targets in tqdm(loader, desc='Training'):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, losses = model(inputs, targets, n_classes, return_losses=True)
        
        # Main classification loss
        task_loss = F.cross_entropy(outputs, targets)
        
        total_loss_batch = (0.5 * task_loss + 
                   0.1 * losses['balance_loss'] + 
                   0.2 * losses['diversity_loss'] + 
                   0.2 * losses['routing_loss'])
        
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += total_loss_batch.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
        # Track individual losses
        loss_components['task'] += task_loss.item()
        for name, loss in losses.items():
            loss_components[name] += loss.item()
    
    # Average losses
    avg_losses = {name: value / len(loader) for name, value in loss_components.items()}
    print("\nLoss components:")
    for name, value in avg_losses.items():
        print(f"{name}: {value:.4f}")
    
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
        num_experts=10,
        expert_hidden_dim=128,
        temp=10.0,
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
    total_epochs = 200
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
    n_classes = 10
    for epoch in range(total_epochs):  
        print(f'\nEpoch: {epoch+1}')
        train_loss, train_acc = train_epoch(
            model, trainloader, optimizer, scheduler,n_classes, device
        )
        
        if epoch % 10 == 0:  
            test_acc = evaluate(model, testloader, device)
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')
            print(f'Test Acc: {test_acc:.3f}%')            
            visualize_moe_expert_map(model, testloader, device,save_to_disk=True,save_path='./plots_aloss')
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Best Test Acc: {best_acc:.3f}%')
    

if __name__ == '__main__':
    main()