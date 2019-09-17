# YOLO.jl - Work in progress!

Currently only supports loading YOLOv2-tiny and the VOC pretrained (by Darknet) model.

Loading of pretrainedMade possible by Yavuz Bakman's work in https://github.com/Ybakman/YoloV2


### Loading a Dataset
```julia
using YOLO

voc = YOLO.datasets.VOC.populate()
sets = YOLO.Settings(image_shape=(416,416),image_channels=3)
vocloaded = YOLO.load(voc, sets)
```
