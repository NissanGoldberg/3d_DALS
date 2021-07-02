# 3D Dals

### Getting Started
download **PyTorch** and **CUDA**

run ***main.py***

### NN model

![](https://github.com/ahatamiz/dals/raw/master/Images/DALS_Framework.png)

### Dataset

download [carvana dataset](https://www.kaggle.com/c/carvana-image-masking-challenge/data) then resize all images to 256x256x1 (grayscale) 
using ***resizer.py*** I wrote in the utils dir

### TODO
* implement **level set and narrow band**,  look in the *resources/level_set_and_narrow_band* dir
* build training, testing of images
* add a 3rd dimension

### Fixes

* refactor code (less hardcoding)

To print the model we can use

### Useful tips

```python
from torchsummary import summary

model = UNET(in_channels=1, out_channels=1)
summary(model, (1, 572, 572))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 572, 572]             576
       BatchNorm2d-2         [-1, 64, 572, 572]             128
              ReLU-3         [-1, 64, 572, 572]               0
            Conv2d-4         [-1, 64, 572, 572]          36,864
       BatchNorm2d-5         [-1, 64, 572, 572]             128
       ...
```

### Further Research
 Look into: 
 * An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale 
    * instead of cnns, making it 3d
    * Instead on UNet or using also UNet
 *  Transformers for Image Recognition 3D/2D segmentation
    * with **adaptive** segmentation
    * According to the paper only with **huge** datasets will this achieve better accuracy
 * From PyTorch to PyTorch Lightning - When scaling   


### Links

* [Deep Active Lesion Segmentation Github](https://github.com/ahatamiz/dals)
    * tensorflow 1.x implementation
* [Arvix reseach paper](https://arxiv.org/pdf/1908.06933.pdf)    
* [Curated Breast Imaging Subset of DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
    * 163.6GB dataset
