# YOLO.jl - Work in progress!
Implementation of YOLO Object Detection models in Julia
Based on https://pjreddie.com/darknet/yolo/

## Known issues

1) The darknet version of yolov2 uses asymmetrical padding, resulting in an output that is `13×13×125×1` rather than `12×12×125×1` that this model gives. That means that pretrained models can't be imported. But I'm exploring training this `12×12×125×1` output from scratch
https://devtalk.nvidia.com/default/topic/1037574/announcements/yolo-object-detection-plugin-for-deepstream-2-0/



## Downloading a dataset
Datasets are not included in this package due to their size, but can be downloaded using the provided download scripts.

Coco (~21 GB of zip files):

`include(joinpath(dirname(dirname(pathof(YOLO))),"datasets/coco/get_coco_dataset.jl"))`


## Initialize with a backend (Flux in progress. Knet pending..)
```
using YOLO
YOLO.LoadBackendHandlers("Flux")
```

## Load a training model
```
YOLOdir = dirname(dirname(pathof(YOLO)))
m = include(joinpath(YOLOdir,"models/yolov2-tiny/trainingmodel.jl"))
```

## Load and prepare an image
```
using Images
imfile = string(@__DIR__,"examples/COCO_train2014_000000000650.jpg")
im, img_size, img_originalsize, padding = YOLO.loadprepareimage(imfile,(416,416)) #Loads, pads and resizes image
im_input = Array{Float32}(undef,416,416,3,1)
im_input[:,:,:,1] = permutedims(collect(channelview(im)),[3,2,1]);
numClasses = 20
summary(im_input)
```
"416×416×3×1 Array{Float32,4}"

## Run model
```
output = m(im_input)
summary(output)
```
"Tracked 12×12×125×1 Array{Float32,4}"

## Loss function
https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088

### IoU
Intersection of Union (IoU) can be 
determined by `bbox_iou` either in the form `bbox_iou(bbox1,bbox2,xywh=true)` 
where bbox1&2 are in the form xywh. Or as a single 1D array of bbox, where the
output is a triangular matrix of the unique and non-self comparisons of the array
