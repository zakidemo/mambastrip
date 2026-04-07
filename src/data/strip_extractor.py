import torch

def image_to_strips(image, patch_width):
    """
    image: [C, H, W]
    return: [num_strips, C, H, patch_width]
    """
    C, H, W = image.shape
    
    strips = []
    
    for i in range(0, W, patch_width):
        strip = image[:, :, i:i+patch_width]
        strips.append(strip)
    
    return torch.stack(strips)