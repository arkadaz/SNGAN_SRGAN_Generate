import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np

device = 'cuda'
gen48 = torch.jit.load(
    r"E:\\FROM_M2\\Work\\Temp\\VR20\\46x46 save model trace\\Generator_46x46.pth", map_location=device)
gen48_to_150 = torch.jit.load(
    r"E:\\FROM_M2\\Work\\Temp\\VR20\\low46_high150 save model trace\\46to150.pth", map_location=device)


def save_tensor_images(image_tensor):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_save = (image_unflat.detach().cpu().squeeze().permute(
        1, 2, 0).numpy()*255).astype(np.uint8)
    print(image_save.shape)
    im = Image.fromarray(image_save)
    num = random.randint(0, 100000000)
    im.save(r"E:\\FROM_M2\\Work\\Temp\\VR20\\Image_gen\\{}.jpeg".format(num))


def get_noise(n_samples, z_dim, device='cuda'):
    return torch.randn(n_samples, z_dim, device=device)


for i in range(0, 10):
    z_dim = 128
    noise = get_noise(1, z_dim, device=device)
    img48_L = gen48(noise)
    save_tensor_images(gen48_to_150(torch.cat((img48_L, img48_L, img48_L), 1)))
