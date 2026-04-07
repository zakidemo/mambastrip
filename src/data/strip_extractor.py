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

def strips_to_image(strips):
    """
    strips: [N, C, H, W]
    return: [C, H, W_total]
    """
    return torch.cat(list(strips), dim=2)

def image_to_horizontal_strips(image, patch_height):
    """
    image: [C, H, W]
    return: [num_strips, C, patch_height, W]
    """
    C, H, W = image.shape
    
    strips = []
    
    for i in range(0, H, patch_height):
        strip = image[:, i:i+patch_height, :]
        strips.append(strip)
    
    return torch.stack(strips)