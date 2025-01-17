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