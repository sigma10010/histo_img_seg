# training



# weights on Cam

python test_wsi.py --model_type='UNet' --fold 1 --loss_type='nll' --n_skip 3
# python test_wsi.py --model_type='UNet' --fold 2 --loss_type='nll' --n_skip 3
# python test_wsi.py --model_type='UNet' --fold 3 --loss_type='nll' --n_skip 3
# python test_wsi.py --model_type='UNet' --fold 4 --loss_type='nll' --n_skip 3
# python test_wsi.py --model_type='UNet' --fold 5 --loss_type='nll' --n_skip 3

# python test_wsi.py --model_type='SE-SSU_Net' --fold 1 --loss_type='nll' --n_head 2 --att_mode='se' --reduction_ratio 8
# python test_wsi.py --model_type='SE-SSU_Net' --fold 2 --loss_type='nll' --n_head 2 --att_mode='se' --reduction_ratio 8
# python test_wsi.py --model_type='SE-SSU_Net' --fold 3 --loss_type='nll' --n_head 2 --att_mode='se' --reduction_ratio 8
# python test_wsi.py --model_type='SE-SSU_Net' --fold 4 --loss_type='nll' --n_head 2 --att_mode='se' --reduction_ratio 8
# python test_wsi.py --model_type='SE-SSU_Net' --fold 5 --loss_type='nll' --n_head 2 --att_mode='se' --reduction_ratio 8

# python test_wsi.py --model_type='SC-SSU_Net' --fold 1 --loss_type='nll' --n_head 2 --is_shortcut '1'
# python test_wsi.py --model_type='SC-SSU_Net' --fold 2 --loss_type='nll' --n_head 2 --is_shortcut '1'
# python test_wsi.py --model_type='SC-SSU_Net' --fold 3 --loss_type='nll' --n_head 2 --is_shortcut '1'
# python test_wsi.py --model_type='SC-SSU_Net' --fold 4 --loss_type='nll' --n_head 2 --is_shortcut '1'
# python test_wsi.py --model_type='SC-SSU_Net' --fold 5 --loss_type='nll' --n_head 2 --is_shortcut '1'

# python test_wsi_2020.py --model_type='SSU_Net' --fold 1 --loss_type='nll+ssim' --n_head 2
# python test_wsi_2020.py --model_type='SSU_Net' --fold 2 --loss_type='nll+ssim' --n_head 2
# python test_wsi_2020.py --model_type='SSU_Net' --fold 3 --loss_type='nll+ssim' --n_head 2
python test_wsi_2020.py --model_type='SSU_Net' --fold 4 --loss_type='nll+ssim' --n_head 2
# python test_wsi_2020.py --model_type='SSU_Net' --fold 5 --loss_type='nll+ssim' --n_head 2

python test_wsi.py --model_type='SSU_Net' --fold 1 --loss_type='nll' --n_head 2
# python test_wsi.py --model_type='SSU_Net' --fold 2 --loss_type='nll' --n_head 2
# python test_wsi.py --model_type='SSU_Net' --fold 3 --loss_type='nll' --n_head 2
# python test_wsi.py --model_type='SSU_Net' --fold 4 --loss_type='nll' --n_head 2
# python test_wsi.py --model_type='SSU_Net' --fold 5 --loss_type='nll' --n_head 2

# python test_wsi_2020.py --model_type='SSU_Net' --fold 1 --loss_type='nll' --n_head 3
# python test_wsi_2020.py --model_type='SSU_Net' --fold 2 --loss_type='nll' --n_head 3
# python test_wsi_2020.py --model_type='SSU_Net' --fold 3 --loss_type='nll' --n_head 3
# python test_wsi_2020.py --model_type='SSU_Net' --fold 4 --loss_type='nll' --n_head 3
# python test_wsi_2020.py --model_type='SSU_Net' --fold 5 --loss_type='nll' --n_head 3

# python test_wsi_2020.py --model_type='SCU_Net' --fold 1 --loss_type='nll' 
# python test_wsi_2020.py --model_type='SCU_Net' --fold 2 --loss_type='nll' 
# python test_wsi_2020.py --model_type='SCU_Net' --fold 3 --loss_type='nll' 
# python test_wsi_2020.py --model_type='SCU_Net' --fold 4 --loss_type='nll' 
# python test_wsi_2020.py --model_type='SCU_Net' --fold 5 --loss_type='nll'

# python test_wsi_2020.py --model_type='AttU_Net' --fold 1 --loss_type='nll' 
# python test_wsi_2020.py --model_type='AttU_Net' --fold 2 --loss_type='nll' 
# python test_wsi_2020.py --model_type='AttU_Net' --fold 3 --loss_type='nll' 
# python test_wsi_2020.py --model_type='AttU_Net' --fold 4 --loss_type='nll' 
# python test_wsi_2020.py --model_type='AttU_Net' --fold 5 --loss_type='nll'

# python test_wsi_2020.py --model_type='CBAMU_Net' --fold 1 --loss_type='nll' --att_mode='cbam'
# python test_wsi_2020.py --model_type='CBAMU_Net' --fold 2 --loss_type='nll' --att_mode='cbam'
# python test_wsi_2020.py --model_type='CBAMU_Net' --fold 3 --loss_type='nll' --att_mode='cbam'
# python test_wsi_2020.py --model_type='CBAMU_Net' --fold 4 --loss_type='nll' --att_mode='cbam'
# python test_wsi_2020.py --model_type='CBAMU_Net' --fold 5 --loss_type='nll' --att_mode='cbam'

# python test_wsi.py --model_type='MHU_Net' --fold 1 --loss_type='nll+ssim' --n_head 3
# python test_wsi.py --model_type='MHU_Net' --fold 2 --loss_type='nll+ssim' --n_head 3
# python test_wsi.py --model_type='MHU_Net' --fold 3 --loss_type='nll+ssim' --n_head 3
# python test_wsi.py --model_type='MHU_Net' --fold 4 --loss_type='nll+ssim' --n_head 3
# python test_wsi.py --model_type='MHU_Net' --fold 5 --loss_type='nll+ssim' --n_head 3

# python test_wsi.py --model_type='CBAMU_Net' --fold 1 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='CBAMU_Net' --fold 2 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='CBAMU_Net' --fold 3 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='CBAMU_Net' --fold 4 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='CBAMU_Net' --fold 5 --loss_type='nll+ssim'

# python test_wsi.py --model_type='MHU_Net' --fold 1 --loss_type='nll' --n_head 4
# python test_wsi.py --model_type='MHU_Net' --fold 2 --loss_type='nll' --n_head 4
# python test_wsi.py --model_type='MHU_Net' --fold 3 --loss_type='nll' --n_head 4
# python test_wsi.py --model_type='MHU_Net' --fold 4 --loss_type='nll' --n_head 4
# python test_wsi.py --model_type='MHU_Net' --fold 5 --loss_type='nll' --n_head 4

# python test_wsi_2020.py --model_type='UNet' --fold 1 --loss_type='nll' 
# python test_wsi_2020.py --model_type='UNet' --fold 2 --loss_type='nll' 
# python test_wsi_2020.py --model_type='UNet' --fold 3 --loss_type='nll' 
python test_wsi_2020.py --model_type='UNet' --fold 4 --loss_type='nll' 
# python test_wsi_2020.py --model_type='UNet' --fold 5 --loss_type='nll' 

# python test_wsi.py --model_type='MHU_Net' --fold 1 --loss_type='nll' --n_head 2
# python test_wsi.py --model_type='MHU_Net' --fold 2 --loss_type='nll' --n_head 2
# python test_wsi.py --model_type='MHU_Net' --fold 3 --loss_type='nll' --n_head 2
# python test_wsi.py --model_type='MHU_Net' --fold 4 --loss_type='nll' --n_head 2
# python test_wsi.py --model_type='MHU_Net' --fold 5 --loss_type='nll' --n_head 2

# python test_wsi_2020.py --model_type='SKU_Net' --fold 1 --loss_type='nll' 
# python test_wsi_2020.py --model_type='SKU_Net' --fold 2 --loss_type='nll' 
# python test_wsi_2020.py --model_type='SKU_Net' --fold 3 --loss_type='nll' 
# python test_wsi_2020.py --model_type='SKU_Net' --fold 4 --loss_type='nll' 
# python test_wsi_2020.py --model_type='SKU_Net' --fold 5 --loss_type='nll' 

# python test_wsi.py --model_type='SKU_Net' --fold 1 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='SKU_Net' --fold 2 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='SKU_Net' --fold 3 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='SKU_Net' --fold 4 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='SKU_Net' --fold 5 --loss_type='nll+ssim' 

# python test_wsi.py --model_type='SK-SE-U_Net' --fold 1 --loss_type='nll' 
# python test_wsi.py --model_type='SK-SE-U_Net' --fold 2 --loss_type='nll' 
# python test_wsi.py --model_type='SK-SE-U_Net' --fold 3 --loss_type='nll' 
# python test_wsi.py --model_type='SK-SE-U_Net' --fold 4 --loss_type='nll' 
# python test_wsi.py --model_type='SK-SE-U_Net' --fold 5 --loss_type='nll' 

# python test_wsi.py --model_type='SKU_Net' --fold 1 --loss_type='nll' 
# python test_wsi.py --model_type='SKU_Net' --fold 2 --loss_type='nll' 
# python test_wsi.py --model_type='SKU_Net' --fold 3 --loss_type='nll' 
# python test_wsi.py --model_type='SKU_Net' --fold 4 --loss_type='nll' 
# python test_wsi.py --model_type='SKU_Net' --fold 5 --loss_type='nll' 

# python test_wsi.py --model_type='UNet' --fold 1 --loss_type='nll' --n_skip 2
# python test_wsi.py --model_type='UNet' --fold 2 --loss_type='nll' --n_skip 2
# python test_wsi.py --model_type='UNet' --fold 3 --loss_type='nll' --n_skip 2
# python test_wsi.py --model_type='UNet' --fold 4 --loss_type='nll' --n_skip 2
# python test_wsi.py --model_type='UNet' --fold 5 --loss_type='nll' --n_skip 2

# python test_wsi.py --model_type='UNet' --fold 1 --loss_type='nll' --width 48
# python test_wsi.py --model_type='UNet' --fold 2 --loss_type='nll' --width 48
# python test_wsi.py --model_type='UNet' --fold 3 --loss_type='nll' --width 48
# python test_wsi.py --model_type='UNet' --fold 4 --loss_type='nll' --width 48
# python test_wsi.py --model_type='UNet' --fold 5 --loss_type='nll' --width 48

# python test_wsi.py --model_type='U_Net' --fold 1 --loss_type='ssim' 
# python test_wsi.py --model_type='U_Net' --fold 2 --loss_type='ssim' 
# python test_wsi.py --model_type='U_Net' --fold 3 --loss_type='ssim' 
# python test_wsi.py --model_type='U_Net' --fold 4 --loss_type='ssim' 
# python test_wsi.py --model_type='U_Net' --fold 5 --loss_type='ssim' 

# python test_wsi.py --model_type='UNet' --fold 1 --loss_type='nll' --depth 3
# python test_wsi.py --model_type='UNet' --fold 2 --loss_type='nll' --depth 3
# python test_wsi.py --model_type='UNet' --fold 3 --loss_type='nll' --depth 3
# python test_wsi.py --model_type='UNet' --fold 4 --loss_type='nll' --depth 3
# python test_wsi.py --model_type='UNet' --fold 5 --loss_type='nll' --depth 3

# python test_wsi.py --model_type='U_Net' --fold 1 --loss_type='nll' --image_size 512 
# python test_wsi.py --model_type='U_Net' --fold 2 --loss_type='nll' --image_size 512 
# python test_wsi.py --model_type='U_Net' --fold 3 --loss_type='nll' --image_size 512 
# python test_wsi.py --model_type='U_Net' --fold 4 --loss_type='nll' --image_size 512
# python test_wsi.py --model_type='U_Net' --fold 5 --loss_type='nll' --image_size 512

# python test_wsi.py --model_type='U_Net' --fold 1 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='U_Net' --fold 2 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='U_Net' --fold 3 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='U_Net' --fold 4 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='U_Net' --fold 5 --loss_type='nll+ssim' 

# python test_wsi.py --model_type='U_Net' --fold 1 --loss_type='nll+ssim+iou' --width 512
# python test_wsi.py --model_type='U_Net' --fold 2 --loss_type='nll+ssim+iou' --width 512
# python test_wsi.py --model_type='U_Net' --fold 3 --loss_type='nll+ssim+iou' --width 512
# python test_wsi.py --model_type='U_Net' --fold 4 --loss_type='nll+ssim+iou' --width 512
# python test_wsi.py --model_type='U_Net' --fold 5 --loss_type='nll+ssim+iou' --width 512

# python test_wsi.py --model_type='U_Net' --fold 1 --loss_type='iou' --alpha 2.0 --gamma 1.0
# python test_wsi.py --model_type='U_Net' --fold 2 --loss_type='iou' --alpha 2.0 --gamma 1.0
# python test_wsi.py --model_type='U_Net' --fold 3 --loss_type='iou' --alpha 2.0 --gamma 1.0
# python test_wsi.py --model_type='U_Net' --fold 4 --loss_type='iou' --alpha 2.0 --gamma 1.0
# python test_wsi.py --model_type='U_Net' --fold 5 --loss_type='iou' --alpha 2.0 --gamma 1.0

# python test_wsi.py --model_type='CBAMU_Net' --fold 1 --loss_type='nll' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='CBAMU_Net' --fold 2 --loss_type='nll' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='CBAMU_Net' --fold 3 --loss_type='nll' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='CBAMU_Net' --fold 4 --loss_type='nll' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='CBAMU_Net' --fold 5 --loss_type='nll' --alpha 2.0 --gamma 1.5




# weights on Titan

# python test_wsi.py --model_type='SK-SSU_Net' --fold 1 --loss_type='nll' --n_head 2 --conv_type='sk'
# python test_wsi.py --model_type='SK-SSU_Net' --fold 2 --loss_type='nll' --n_head 2 --conv_type='sk'
# python test_wsi.py --model_type='SK-SSU_Net' --fold 3 --loss_type='nll' --n_head 2 --conv_type='sk'
# python test_wsi.py --model_type='SK-SSU_Net' --fold 4 --loss_type='nll' --n_head 2 --conv_type='sk'
# python test_wsi.py --model_type='SK-SSU_Net' --fold 5 --loss_type='nll' --n_head 2 --conv_type='sk'

python test_wsi.py --model_type='SSU_Net' --fold 1 --loss_type='nll+ssim' --n_head 2
# python test_wsi.py --model_type='SSU_Net' --fold 2 --loss_type='nll+ssim' --n_head 2
# python test_wsi.py --model_type='SSU_Net' --fold 3 --loss_type='nll+ssim' --n_head 2
# python test_wsi.py --model_type='SSU_Net' --fold 4 --loss_type='nll+ssim' --n_head 2
# python test_wsi.py --model_type='SSU_Net' --fold 5 --loss_type='nll+ssim' --n_head 2


# python test_wsi_2020.py --model_type='SSU_Net' --fold 1 --loss_type='nll' --n_head 2
# python test_wsi_2020.py --model_type='SSU_Net' --fold 2 --loss_type='nll' --n_head 2
# python test_wsi_2020.py --model_type='SSU_Net' --fold 3 --loss_type='nll' --n_head 2
python test_wsi_2020.py --model_type='SSU_Net' --fold 4 --loss_type='nll' --n_head 2
# python test_wsi_2020.py --model_type='SSU_Net' --fold 5 --loss_type='nll' --n_head 2

# python test_wsi.py --model_type='SSU_Net' --fold 1 --loss_type='nll' --n_head 3
# python test_wsi.py --model_type='SSU_Net' --fold 2 --loss_type='nll' --n_head 3
# python test_wsi.py --model_type='SSU_Net' --fold 3 --loss_type='nll' --n_head 3
# python test_wsi.py --model_type='SSU_Net' --fold 4 --loss_type='nll' --n_head 3
# python test_wsi.py --model_type='SSU_Net' --fold 5 --loss_type='nll' --n_head 3

# python test_wsi_2020.py --model_type='UNet' --fold 1 --loss_type='nll' --n_skip 3
# python test_wsi_2020.py --model_type='UNet' --fold 2 --loss_type='nll' --n_skip 3
# python test_wsi_2020.py --model_type='UNet' --fold 3 --loss_type='nll' --n_skip 3
# python test_wsi_2020.py --model_type='UNet' --fold 4 --loss_type='nll' --n_skip 3
# python test_wsi_2020.py --model_type='UNet' --fold 5 --loss_type='nll' --n_skip 3

# python test_wsi.py --model_type='SCU_Net' --fold 1 --loss_type='nll' 
# python test_wsi.py --model_type='SCU_Net' --fold 2 --loss_type='nll' 
# python test_wsi.py --model_type='SCU_Net' --fold 3 --loss_type='nll' 
# python test_wsi.py --model_type='SCU_Net' --fold 4 --loss_type='nll' 
# python test_wsi.py --model_type='SCU_Net' --fold 5 --loss_type='nll'

# python test_wsi_2020.py --model_type='BAMU_Net' --fold 1 --loss_type='nll' --att_mode='bam'
# python test_wsi_2020.py --model_type='BAMU_Net' --fold 2 --loss_type='nll' --att_mode='bam'
# python test_wsi_2020.py --model_type='BAMU_Net' --fold 3 --loss_type='nll' --att_mode='bam'
# python test_wsi_2020.py --model_type='BAMU_Net' --fold 4 --loss_type='nll' --att_mode='bam'
# python test_wsi_2020.py --model_type='BAMU_Net' --fold 5 --loss_type='nll' --att_mode='bam'

# python test_wsi_2020.py --model_type='MHU_Net' --fold 1 --loss_type='nll+ssim' --n_head 3
# python test_wsi_2020.py --model_type='MHU_Net' --fold 2 --loss_type='nll+ssim' --n_head 3
# python test_wsi_2020.py --model_type='MHU_Net' --fold 3 --loss_type='nll+ssim' --n_head 3
# python test_wsi_2020.py --model_type='MHU_Net' --fold 4 --loss_type='nll+ssim' --n_head 3
# python test_wsi_2020.py --model_type='MHU_Net' --fold 5 --loss_type='nll+ssim' --n_head 3

# python test_wsi_2020.py --model_type='MHU_Net' --fold 1 --loss_type='nll' --n_head 3
# python test_wsi_2020.py --model_type='MHU_Net' --fold 2 --loss_type='nll' --n_head 3
# python test_wsi_2020.py --model_type='MHU_Net' --fold 3 --loss_type='nll' --n_head 3
# python test_wsi_2020.py --model_type='MHU_Net' --fold 4 --loss_type='nll' --n_head 3
# python test_wsi_2020.py --model_type='MHU_Net' --fold 5 --loss_type='nll' --n_head 3

# python test_wsi.py --model_type='BAMU_Net' --fold 1 --loss_type='nll+ssim' --att_mode='bam'
# python test_wsi.py --model_type='BAMU_Net' --fold 2 --loss_type='nll+ssim' --att_mode='bam'
# python test_wsi.py --model_type='BAMU_Net' --fold 3 --loss_type='nll+ssim' --att_mode='bam'
# python test_wsi.py --model_type='BAMU_Net' --fold 4 --loss_type='nll+ssim' --att_mode='bam'
# python test_wsi.py --model_type='BAMU_Net' --fold 5 --loss_type='nll+ssim' --att_mode='bam'

# python test_wsi_2020.py --model_type='SEU_Net' --fold 1 --loss_type='nll' --att_mode='se'
# python test_wsi_2020.py --model_type='SEU_Net' --fold 2 --loss_type='nll' --att_mode='se'
# python test_wsi_2020.py --model_type='SEU_Net' --fold 3 --loss_type='nll' --att_mode='se'
# python test_wsi_2020.py --model_type='SEU_Net' --fold 4 --loss_type='nll' --att_mode='se'
# python test_wsi_2020.py --model_type='SEU_Net' --fold 5 --loss_type='nll' --att_mode='se'

# python test_wsi.py --model_type='MHU_Net' --fold 1 --loss_type='nll' --n_head 3
# python test_wsi.py --model_type='MHU_Net' --fold 2 --loss_type='nll' --n_head 3
# python test_wsi.py --model_type='MHU_Net' --fold 3 --loss_type='nll' --n_head 3
# python test_wsi.py --model_type='MHU_Net' --fold 4 --loss_type='nll' --n_head 3
# python test_wsi.py --model_type='MHU_Net' --fold 5 --loss_type='nll' --n_head 3

# python test_wsi.py --model_type='SEU_Net' --fold 1 --loss_type='nll+ssim' --att_mode='se'
# python test_wsi.py --model_type='SEU_Net' --fold 2 --loss_type='nll+ssim' --att_mode='se'
# python test_wsi.py --model_type='SEU_Net' --fold 3 --loss_type='nll+ssim' --att_mode='se'
# python test_wsi.py --model_type='SEU_Net' --fold 4 --loss_type='nll+ssim' --att_mode='se'
# python test_wsi.py --model_type='SEU_Net' --fold 5 --loss_type='nll+ssim' --att_mode='se'

# python test_wsi.py --model_type='AttU_Net' --fold 1 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='AttU_Net' --fold 2 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='AttU_Net' --fold 3 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='AttU_Net' --fold 4 --loss_type='nll+ssim' 
# python test_wsi.py --model_type='AttU_Net' --fold 5 --loss_type='nll+ssim' 

# python test_wsi.py --model_type='SK-SC-U_Net' --fold 1 --loss_type='nll' 
# python test_wsi.py --model_type='SK-SC-U_Net' --fold 2 --loss_type='nll' 
# python test_wsi.py --model_type='SK-SC-U_Net' --fold 3 --loss_type='nll' 
# python test_wsi.py --model_type='SK-SC-U_Net' --fold 4 --loss_type='nll' 
# python test_wsi.py --model_type='SK-SC-U_Net' --fold 5 --loss_type='nll' 

# python test_wsi.py --model_type='UNet' --fold 1 --loss_type='nll' --n_skip 1
# python test_wsi.py --model_type='UNet' --fold 2 --loss_type='nll' --n_skip 1
# python test_wsi.py --model_type='UNet' --fold 3 --loss_type='nll' --n_skip 1
# python test_wsi.py --model_type='UNet' --fold 4 --loss_type='nll' --n_skip 1
# python test_wsi.py --model_type='UNet' --fold 5 --loss_type='nll' --n_skip 1

# python test_wsi.py --model_type='UNet' --fold 1 --loss_type='nll' --n_skip 3
# python test_wsi.py --model_type='UNet' --fold 2 --loss_type='nll' --n_skip 3
# python test_wsi.py --model_type='UNet' --fold 3 --loss_type='nll' --n_skip 3
# python test_wsi.py --model_type='UNet' --fold 4 --loss_type='nll' --n_skip 3
# python test_wsi.py --model_type='UNet' --fold 5 --loss_type='nll' --n_skip 3

# python test_wsi.py --model_type='UNet' --fold 1 --loss_type='nll' --n_skip 0
# python test_wsi.py --model_type='UNet' --fold 2 --loss_type='nll' --n_skip 0
# python test_wsi.py --model_type='UNet' --fold 3 --loss_type='nll' --n_skip 0
# python test_wsi.py --model_type='UNet' --fold 4 --loss_type='nll' --n_skip 0
# python test_wsi.py --model_type='UNet' --fold 5 --loss_type='nll' --n_skip 0

# python test_wsi.py --model_type='UNet' --fold 1 --loss_type='nll' --width 16
# python test_wsi.py --model_type='UNet' --fold 2 --loss_type='nll' --width 16
# python test_wsi.py --model_type='UNet' --fold 3 --loss_type='nll' --width 16
# python test_wsi.py --model_type='UNet' --fold 4 --loss_type='nll' --width 16
# python test_wsi.py --model_type='UNet' --fold 5 --loss_type='nll' --width 16

# python test_wsi.py --model_type='UNet' --fold 1 --loss_type='nll' --depth 4
# python test_wsi.py --model_type='UNet' --fold 2 --loss_type='nll' --depth 4
# python test_wsi.py --model_type='UNet' --fold 3 --loss_type='nll' --depth 4
# python test_wsi.py --model_type='UNet' --fold 4 --loss_type='nll' --depth 4
# python test_wsi.py --model_type='UNet' --fold 5 --loss_type='nll' --depth 4

# python test_wsi.py --model_type='BAMU_Net' --fold 1 --loss_type='nll' --alpha 2.0 --gamma 1.0 --att_mode='bam'
# python test_wsi.py --model_type='BAMU_Net' --fold 2 --loss_type='nll' --alpha 2.0 --gamma 1.0 --att_mode='bam'
# python test_wsi.py --model_type='BAMU_Net' --fold 3 --loss_type='nll' --alpha 2.0 --gamma 1.0 --att_mode='bam'
# python test_wsi.py --model_type='BAMU_Net' --fold 4 --loss_type='nll' --alpha 2.0 --gamma 1.0 --att_mode='bam'
# python test_wsi.py --model_type='BAMU_Net' --fold 5 --loss_type='nll' --alpha 2.0 --gamma 1.0 --att_mode='bam'

# python test_wsi.py --model_type='SEU_Net' --fold 1 --loss_type='nll' --alpha 2.0 --gamma 1.5 --att_mode='se'
# python test_wsi.py --model_type='SEU_Net' --fold 2 --loss_type='nll' --alpha 2.0 --gamma 1.5 --att_mode='se'
# python test_wsi.py --model_type='SEU_Net' --fold 3 --loss_type='nll' --alpha 2.0 --gamma 1.5 --att_mode='se'
# python test_wsi.py --model_type='SEU_Net' --fold 4 --loss_type='nll' --alpha 2.0 --gamma 1.5 --att_mode='se'
# python test_wsi.py --model_type='SEU_Net' --fold 5 --loss_type='nll' --alpha 2.0 --gamma 1.5 --att_mode='se'

# python test_wsi.py --model_type='U_Net' --fold 1 --loss_type='l1' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='U_Net' --fold 2 --loss_type='l1' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='U_Net' --fold 3 --loss_type='l1' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='U_Net' --fold 4 --loss_type='l1' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='U_Net' --fold 5 --loss_type='l1' --alpha 2.0 --gamma 1.5

# python test_wsi.py --model_type='AttU_Net' --fold 1 --loss_type='nll' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='AttU_Net' --fold 2 --loss_type='nll' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='AttU_Net' --fold 3 --loss_type='nll' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='AttU_Net' --fold 4 --loss_type='nll' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='AttU_Net' --fold 5 --loss_type='nll' --alpha 2.0 --gamma 1.5

# python test_wsi.py --model_type='U_Net' --fold 1 --loss_type='nll' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='U_Net' --fold 2 --loss_type='nll' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='U_Net' --fold 3 --loss_type='nll' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='U_Net' --fold 4 --loss_type='nll' --alpha 2.0 --gamma 1.5
# python test_wsi.py --model_type='U_Net' --fold 5 --loss_type='nll' --alpha 2.0 --gamma 1.5

# python test_wsi.py --model_type='U_Net' --fold 1 --loss_type='nll' --image_size 256 --width 512
# python test_wsi.py --model_type='U_Net' --fold 2 --loss_type='nll' --image_size 256 --width 512
# python test_wsi.py --model_type='U_Net' --fold 3 --loss_type='nll' --image_size 256 --width 512
# python test_wsi.py --model_type='U_Net' --fold 4 --loss_type='nll' --image_size 256 --width 512
# python test_wsi.py --model_type='U_Net' --fold 5 --loss_type='nll' --image_size 256 --width 512

# python test_wsi.py --model_type='U_Net' --fold 1 --loss_type='nll+iou' --width 512
# python test_wsi.py --model_type='U_Net' --fold 2 --loss_type='nll+iou' --width 512
# python test_wsi.py --model_type='U_Net' --fold 3 --loss_type='nll+iou' --width 512
# python test_wsi.py --model_type='U_Net' --fold 4 --loss_type='nll+iou' --width 512
# python test_wsi.py --model_type='U_Net' --fold 5 --loss_type='nll+iou' --width 512