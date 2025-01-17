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
        self.temperature = temperature
        
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
    def __init__(self, num_experts=8, expert_hidden_dim=256, temp=0.1, diversity_coef=0.1,specialization_coef=0.1):
        super().__init__()
        self.diversity_coef = diversity_coef
        self.specialization_coef = specialization_coef
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
    
    def forward(self, x,labels=None,n_classes = None, return_losses=False):
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
            balance_loss = self.calculate_balance_loss(expert_weights,labels,n_classes)
            specialization_loss = self.calculate_specialization_loss(expert_weights, labels,n_classes)
            
            # Auxiliary losses dictionary
            losses = {
                'balance_loss': balance_loss,
                'routing_loss': specialization_loss

            }
            return output, losses
            
        return output
    
    def calculate_balance_loss(self, expert_weights, labels, num_classes):
        """
        Calculate balance loss based on class distribution in batch
        """
        batch_size = labels.size(0)
        num_experts = expert_weights.size(1)
        
        # Get class distribution in current batch
        class_counts = torch.bincount(labels, minlength=num_classes)
        class_distribution = class_counts.float() / batch_size  # [num_classes]
        
        # Since we have fewer experts than classes, we need to map/group classes to experts
        # Simple approach: average pool the class distribution to match number of experts
        ideal_usage = F.adaptive_avg_pool1d(
            class_distribution.view(1, 1, -1),  # [1, 1, num_classes]
            num_experts                         # target length
        ).squeeze()  # [num_experts]
        
        # Get actual expert usage
        expert_usage = expert_weights.mean(0)  # [num_experts]
        
        # Now both are shape [num_experts]
        balance_loss = F.mse_loss(expert_usage, ideal_usage)
        
        return balance_loss
     
    def calculate_specialization_loss(self, expert_weights, labels, num_classes):
        """
        Encourage each expert to be strongly decisive about which classes it handles
        """
        batch_size = expert_weights.size(0)
        
        # Get class-expert correlations
        labels_onehot = F.one_hot(labels, num_classes).float()
        expert_class_correlation = torch.matmul(
            expert_weights.t(),  # [num_experts, batch_size]
            labels_onehot        # [batch_size, num_classes]
        ) / batch_size          # [num_experts, num_classes]
        
        # We want each expert to have high confidence (close to 0 or 1)
        # for each class - encourage decisive behavior
        decisiveness_loss = -torch.mean(
            torch.abs(expert_class_correlation - 0.5)
        )
        
        return decisiveness_loss

def train_epoch(model, loader, optimizer, scheduler,n_classes, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, desc='Training'):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, aux_losses = model(inputs,targets,n_classes, return_losses=True)
        
        # Main classification loss
        task_loss = F.cross_entropy(outputs, targets)
        
        # Combine all losses
        loss = task_loss + aux_losses['balance_loss'] + aux_losses['routing_loss']
        
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

import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_moe(model, dataloader, device, num_batches=1, save_to_disk=False, save_path=None):
    """
    Simple visualization of expert utilization.
    Args:
        save_to_disk (bool): If True, saves plots to disk instead of displaying
        save_path (str): Directory to save plots (default is current directory)
    """
    model.eval()
    expert_usage = []
    class_expert_usage = {i: [] for i in range(10)}
    cifar_classes = ['plane', 'car', 'bird', 'cat', 'deer', 
                     'dog', 'frog', 'horse', 'ship', 'truck']
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            images, labels = images.to(device), labels.to(device)
            
            # Get features and gate weights
            features = model.backbone(images)[0]
            gate_weights = model.gate(features)
            
            # Track usage
            expert_usage.append(gate_weights.cpu().numpy())
            
            # Track per-class usage
            for label in range(10):
                mask = labels == label
                if mask.any():
                    class_expert_usage[label].append(
                        gate_weights[mask].mean(0).cpu().numpy()
                    )
    
    # Create figure
    fig = plt.figure(figsize=(10, 4))
    
    # Average expert usage
    avg_usage = np.mean(np.concatenate(expert_usage, axis=0), axis=0)
    plt.subplot(1, 2, 1)
    plt.bar(range(len(avg_usage)), avg_usage)
    plt.title('Overall Expert Utilization')
    plt.xlabel('Expert ID')
    plt.ylabel('Average Usage')
    
    # Per-class expert preference
    plt.subplot(1, 2, 2)
    class_expert_map = np.zeros((10, len(avg_usage)))
    for i in range(10):
        if class_expert_usage[i]:
            class_expert_map[i] = np.mean(np.stack(class_expert_usage[i]), axis=0)
    
    plt.imshow(class_expert_map, aspect='auto')
    plt.colorbar(label='Usage')
    plt.yticks(range(10), cifar_classes)
    plt.xlabel('Expert ID')
    plt.title('Class-Expert Preferences')
    
    plt.tight_layout()
    
    if save_to_disk:
        # Create save path if it doesn't exist
        if save_path is None:
            save_path = '.'
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = os.path.join(save_path, f'moe_visualization_{timestamp}.png')
        
        # Save with high DPI for quality
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Plot saved to: {save_file}")
    else:
        plt.show()
        
    # Return the computed statistics for potential further use
    return {
        'expert_usage': avg_usage,
        'class_expert_map': class_expert_map
    }

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
        diversity_coef=0.15,
        specialization_coef = 0.3
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
    n_classes = 10
    for epoch in range(total_epochs):  
        print(f'\nEpoch: {epoch+1}')
        train_loss, train_acc = train_epoch(
            model, trainloader, optimizer, scheduler,n_classes, device
        )
        
        if epoch % 1 == 0:  
            test_acc = evaluate(model, testloader, device)
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')
            print(f'Test Acc: {test_acc:.3f}%')            
            visualize_moe(model, testloader, device,save_to_disk=True,save_path='./plots')
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Best Test Acc: {best_acc:.3f}%')
    

if __name__ == '__main__':
    main()