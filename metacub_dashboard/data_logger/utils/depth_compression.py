import torch
import numpy as np
import gc
import time


def depth2rgb(depth, depth_min, depth_max):
    # colored_depth_aux = hc.depth2rgb(copy.deepcopy(depth.squeeze()), zrange=(d_min, d_max), inv_depth=True)

    d = torch.tensor(depth.squeeze(), device='cuda', requires_grad=False)
    d_min, d_max = 1 / depth_max, 1 / depth_min
    d.reciprocal_()

    d.sub_(d_min)
    d.div_(d_max - d_min)  # d = (d - d_min) / (d_max - d_min)
    d.mul_(5)  # d = d * 300 / 60

    # w = torch.floor(d).to(int) % 6
    w = torch.empty_like(d)
    torch.floor(d, out=w)
    w.remainder_(6)

    d.frac_()  # d = (d - torch.floor(d).to(int))
    one_minus_d = 1 - d

    w_old = w
    w = w_old.to(torch.uint8).cpu()
    d = d.cpu()
    one_minus_d = one_minus_d.cpu()
    del w_old
    torch.cuda.empty_cache()
    gc.collect()

    mask_w0 = (w == 0)
    mask_w1 = (w == 1)
    mask_w2 = (w == 2)
    mask_w3 = (w == 3)
    mask_w4 = (w == 4)
    mask_w5 = (w == 5)

    r = torch.zeros_like(d)
    g = torch.zeros_like(d)
    b = torch.zeros_like(d)

    # r = (w==0) * 1.0 + (w==1) * (1-d) + (w==4) * d + (w==5) * 1.0

    # mask_w0, mask_w1, mask_w4, mask_w5 = mask_w0.cuda(), mask_w1.cuda(), mask_w4.cuda(), mask_w5.cuda()

    r[mask_w0] = 1
    r[mask_w1] = one_minus_d[mask_w1]
    r[mask_w4] = d[mask_w4]
    r[mask_w5] = 1

    # g = (w==0) * d + (w==1) * 1.0 + (w==2) * 1.0 + (w==3) * (1-d) 

    g[mask_w0] = d[mask_w0]
    g[mask_w1] = 1
    g[mask_w2] = 1
    g[mask_w3] = one_minus_d[mask_w3]

    # b = (w==2) * d + (w==3) * 1.0 + (w==4) * 1.0 + (w==5) * (1-d)

    b[mask_w2] = d[mask_w2]
    b[mask_w3] = 1.0
    b[mask_w4] = 1.0
    b[mask_w5] = one_minus_d[mask_w5]

    colored_depth = torch.stack((r, g, b), -1)

    colored_depth = torch.round(colored_depth * 255).cpu().numpy().astype(np.uint8)
    # decompress_depth_data(colored_depth, depth_min, depth_max)
    return colored_depth

def rgb2depth(rgb, depth_min, depth_max):
    # Decompress the depth data
    disp_min = 1 / depth_max
    disp_max = 1 / depth_min

    rgb = torch.tensor(rgb)
    hsv = torch.zeros_like(rgb, pin_memory=True)
    depth = torch.zeros(rgb.shape[:-1], dtype=torch.float32, pin_memory=True)

    batch_size = 1024
    stream = torch.cuda.Stream()

    start_time = time.time()
    with torch.cuda.stream(stream):
        for start in range(0, rgb.shape[0], batch_size):
            end = min(start + batch_size, rgb.shape[0])

            # batch_rgb[:end-start] = rgb[start:end]
            batch_rgb = rgb[start:end].to(device='cuda', non_blocking=True).float()
            hsv_batch = torch.zeros_like(batch_rgb)

            # Dequantize
            batch_rgb.div_(255)

            rgb_max, rgb_amax = batch_rgb.max(-1)
            rgb_min, _ = batch_rgb.min(-1)

            r = (rgb_max - rgb_min).to(torch.float32)

            m = (rgb_amax == 0)
            hsv_batch[...,0] = (0+(batch_rgb[..., 1] - batch_rgb[..., 2]) / r) * m

            m = (rgb_amax == 1)
            hsv_batch[...,0] += (2+(batch_rgb[..., 2] - batch_rgb[..., 0]) / r) * m

            m = (rgb_amax == 2)
            hsv_batch[...,0] += (4+(batch_rgb[..., 0] - batch_rgb[..., 1]) / r) * m
            # fmt: on
            hsv_batch[...,0] *= 60
            hsv_batch[hsv_batch < 0] += 360

            hsv_batch[...,1] = r / rgb_max
            hsv_batch[...,2] = rgb_max

            depth_batch = (hsv_batch[...,0] / 300).clone()
            depth_batch = depth_batch * (disp_max - disp_min) + disp_min
            depth_batch = 1 / depth_batch

            hsv[start:end] = hsv_batch.to('cpu')
            depth[start:end] = depth_batch.to('cpu') 
    return depth.numpy()