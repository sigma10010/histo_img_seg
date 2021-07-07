### whole-slide image segmentation


Implementaion of the adaptive-scale U-Net with attention gate (AG), bottleneck attention module (BAM), convolutional block attention module (CBAM), squeeze-and-exitation (SE) and selective-kernel (SK).

AG: https://arxiv.org/abs/1808.08114

BAM: https://arxiv.org/abs/1807.06514

CBAM: https://arxiv.org/abs/1807.06521

SE: https://arxiv.org/abs/1709.01507?spm=a2c41.13233144.0.0

SK: https://arxiv.org/abs/1903.06586

Suport multi-head prediction, shortcut connection, dynamic feature selection.

Suport cross-entropy loss, strutural similarity (SSIM) loss, Dice loss and IoU loss.

Tested on datasets from PAIP 2019 and PAIP 2020.

## PAIP 2019
![2019](/results/segmap.jpeg)

## PAIP 2020
![2019](/results/segmap2020.jpeg)
