# YOLO.jl

Currently only supports loading [YOLOv2-tiny](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg) and the [VOC-2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) pretrained model (pretrained on [Darknet](https://pjreddie.com/darknet/)).

The majority of this is made possible by Yavuz Bakman's great work in https://github.com/Ybakman/YoloV2

<p float="left">
<img src="examples/boat.png" alt="drawing" width="200"/>
<img src="examples/bikes.png" alt="bikes" width="200"/>
<img src="examples/cowcat.png" alt="cowcat" width="200"/>
<img src="examples/cars.png" alt="cars" width="200"/>
</p>

See below for examples or ask questions on [![Join the julia slack](https://img.shields.io/badge/slack-%23machine--learning-yellow)](https://slackinvite.julialang.org)

| **Platform**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| x86 & AARCH Linux, MacOS | [![][travis-img]][travis-url] |


## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add YOLO
```


## Example Usage (WIP)

### Testing a dataset
```julia
using YOLO

mod = YOLO.v3_tiny_416_COCO()

batch = YOLO.emptybatch(mod)

img = load(joinpath(dirname(dirname(pathof(YOLO))),"test","images","dog-cycle-car.png"))
batch[:,:,:,1] .= YOLO.gpu(resizePadImage(img, mod))

res = mod(batch)

```


[discourse-tag-url]: https://discourse.julialang.org/tags/yolo

[travis-img]: https://travis-ci.com/ianshmean/YOLO.jl.svg?branch=master
[travis-url]: https://travis-ci.com/ianshmean/YOLO.jl

[codecov-img]: https://codecov.io/gh/ianshmean/YOLO.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/ianshmean/YOLO.jl

[coveralls-img]: https://coveralls.io/repos/github/ianshmean/YOLO.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/ianshmean/YOLO.jl?branch=master

[issues-url]: https://github.com/ianshmean/YOLO.jl/issues
