from thop import profile
import torch
from models.u_net import U_Net,R2U_Net,AttU_Net,R2AttU_Net,XXU_Net
from models.unet import UNet, UNet_V1, UNet_V2, UNet_V3, UNet_V4
from models.swin_unet import SwinTransformerSys
from models.transunet import TransUNetWithAttention
from models.swinunet import SwinU

from models.vit_seg_modeling import VisionTransformer as ViT_seg
from models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from ptflops import get_model_complexity_info

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#command: python -m scripts.flops_params

# TransUNet = TransUNetWithAttention(in_ch=3, out_ch=2)
vit_name ='ViT-B_16'
config_vit = CONFIGS_ViT_seg[vit_name]
config_vit.n_classes = 2
config_vit.n_skip = 0
# if vit_name.find('R50') != -1:
    # config_vit.patches.grid = (int(512 / 16), int(512 / 16))
TransUNet = ViT_seg(config_vit)

safs = UNet_V4(n_classes=2, reduction_ratio=None, n_head = 2, att_mode = 'bam', is_scale_selective = True, is_shortcut = True, conv_type = 'basic')
unet = UNet_V4(n_classes=2, reduction_ratio=None, n_head = 1, att_mode = 'bam', is_scale_selective = False, is_shortcut = True, conv_type = 'basic')
se = UNet_V4(n_classes=2, reduction_ratio=8, n_head = 1, att_mode = 'se', is_scale_selective = False, is_shortcut = True, conv_type = 'basic')
cbam = UNet_V4(n_classes=2, reduction_ratio=8, n_head = 1, att_mode = 'cbam', is_scale_selective = False, is_shortcut = True, conv_type = 'basic')
bam = UNet_V4(n_classes=2, reduction_ratio=8, n_head = 1, att_mode = 'bam', is_scale_selective = False, is_shortcut = True, conv_type = 'basic')
sk = UNet_V4(n_classes=2, reduction_ratio=None, n_head = 1, att_mode = 'bam', is_scale_selective = False, is_shortcut = True, conv_type = 'sk')
swin_unet = SwinTransformerSys(img_size=512, num_classes = 2)
swinunet = SwinU()
# sask = UNet_V4(n_classes=2, reduction_ratio=None, n_head = 1, att_mode = 'bam', is_scale_selective = False, is_shortcut = True, conv_type = 'sk')

input_tensor = torch.randn(1, 3, 512, 512)
y,_ = TransUNet(input_tensor)
print(y.size())

# 计算 FLOPs 和参数量
# print('model: unet')
# flops, params = profile(unet, inputs=(input_tensor, ))
# print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
# print(f"Params: {params / 1e6:.2f} M")

# print('model: safs')
# flops, params = profile(safs, inputs=(input_tensor, ))
# print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
# print(f"Params: {params / 1e6:.2f} M")

# print('model: se')
# flops, params = profile(se, inputs=(input_tensor, ))
# print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
# print(f"Params: {params / 1e6:.2f} M")

# print('model: sk')
# flops, params = profile(sk, inputs=(input_tensor, ))
# print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
# print(f"Params: {params / 1e6:.2f} M")

# print('model: cbam')
# flops, params = profile(cbam, inputs=(input_tensor, ))
# print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
# print(f"Params: {params / 1e6:.2f} M")

# print('model: bam')
# flops, params = profile(bam, inputs=(input_tensor, ))
# print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
# print(f"Params: {params / 1e6:.2f} M")

# print('model: swin_unet')
# flops, params = profile(swin_unet, inputs=(input_tensor, ))
# print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
# print(f"Params: {params / 1e6:.2f} M")

input_res = (3, 512, 512)

# print('model: unet')
# macs, params = get_model_complexity_info(unet, input_res, as_strings=True, print_per_layer_stat=False, verbose=False)
# print(f'FLOPs (MACs): {macs}')
# print(f'Params: {params}')

# print('model: safs')
# macs, params = get_model_complexity_info(safs, input_res, as_strings=True, print_per_layer_stat=False, verbose=False)
# print(f'FLOPs (MACs): {macs}')
# print(f'Params: {params}')

# print('model: se')
# macs, params = get_model_complexity_info(se, input_res, as_strings=True, print_per_layer_stat=False, verbose=False)
# print(f'FLOPs (MACs): {macs}')
# print(f'Params: {params}')

# print('model: sk')
# macs, params = get_model_complexity_info(sk, input_res, as_strings=True, print_per_layer_stat=False, verbose=False)
# print(f'FLOPs (MACs): {macs}')
# print(f'Params: {params}')

# print('model: cbam')
# macs, params = get_model_complexity_info(cbam, input_res, as_strings=True, print_per_layer_stat=False, verbose=False)
# print(f'FLOPs (MACs): {macs}')
# print(f'Params: {params}')

# print('model: bam')
# macs, params = get_model_complexity_info(bam, input_res, as_strings=True, print_per_layer_stat=False, verbose=False)
# print(f'FLOPs (MACs): {macs}')
# print(f'Params: {params}')

# print('model: swin_unet')
# macs, params = get_model_complexity_info(swin_unet, input_res, as_strings=True, print_per_layer_stat=False, verbose=False)
# print(f'FLOPs (MACs): {macs}')
# print(f'Params: {params}')

print('model: swinunet')
macs, params = get_model_complexity_info(swinunet, input_res, as_strings=True, print_per_layer_stat=False, verbose=False)
print(f'FLOPs (MACs): {macs}')
print(f'Params: {params}')

print('model: TransUNet')
macs, params = get_model_complexity_info(TransUNet, input_res, as_strings=True, print_per_layer_stat=False, verbose=False)
print(f'FLOPs (MACs): {macs}')
print(f'Params: {params}')


# print(swin_unet.flops())
