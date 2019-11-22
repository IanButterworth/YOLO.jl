# YOLO.jl

YOLO object detection natively in Julia, through loading Darknet .cfg and .weights files as Flux models.
Core functionality based on https://github.com/r3tex/ObjectDetector.jl

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

Requires julia v1.3+

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add YOLO
```


## Example Usage (WIP)

### Loading and running on an image
```julia
using YOLO

mod = YOLO.v3_tiny_416_COCO()

batch = YOLO.emptybatch(mod)

img = load(joinpath(dirname(dirname(pathof(YOLO))),"test","images","dog-cycle-car.png"))
batch[:,:,:,1] .= YOLO.gpu(resizePadImage(img, mod))

res = mod(batch)
```

## Pretrained Models
Most of the darknet models that are pretrained on the COCO dataset are available:
```julia
    YOLO.v2_608_COCO()
    YOLO.v2_tiny_416_COCO()
    YOLO.v3_320_COCO()
    YOLO.v3_416_COCO()
    YOLO.v3_608_COCO()
    YOLO.v3_608_spp_COCO()
    YOLO.v3_tiny_416_COCO()
```

Or custom models can be loaded with:
```julia
Yolo("path/to/model.cfg", "path/to/weights.weights", 1)
```
where `1` is the batch size.

For instance the pretrained models are defined as:
```julia
v2_608_COCO(;batch=1, silent=false) = Yolo(joinpath(models_dir,"yolov2-608.cfg"), getArtifact("yolov2-COCO"), batch, silent=silent)
```

The weights are stored as lazily-loaded julia artifacts.


[discourse-tag-url]: https://discourse.julialang.org/tags/yolo

[travis-img]: https://travis-ci.com/ianshmean/YOLO.jl.svg?branch=master
[travis-url]: https://travis-ci.com/ianshmean/YOLO.jl

[codecov-img]: https://codecov.io/gh/ianshmean/YOLO.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/ianshmean/YOLO.jl

[coveralls-img]: https://coveralls.io/repos/github/ianshmean/YOLO.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/ianshmean/YOLO.jl?branch=master

[issues-url]: https://github.com/ianshmean/YOLO.jl/issues
