# YOLO.jl

Currently only supports loading [YOLOv2-tiny](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg) and the [VOC-2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) pretrained model (pretrained on [Darknet](https://pjreddie.com/darknet/)).

The majority of this is made possible by Yavuz Bakman's great work in https://github.com/Ybakman/YoloV2

**Docs**

See below for examples or ask questions on [![Join the julia slack](https://img.shields.io/badge/slack-%23machine--learning-yellow)](https://slackinvite.julialang.org)

| **Platform**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| Linux & MacOS x86 | [![][travis-img]][travis-url] |
| Windows 32/64-bit | [![][appveyor-img]][appveyor-url] |
| Linux ARM 32/64-bit | [![][drone-img]][drone-url] |
| FreeBSD x86 | [![][cirrus-img]][cirrus-url] |
|  | [![Codecoverage Status][codecov-img]][codecov-url]<br>[![Coveralls Status][coveralls-img]][coveralls-url] |


## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add https://github.com/ianshmean/YOLO.jl
```

## Example Usage (WIP)
```julia
using YOLO

#First time only (downloads 5011 images & labels!)
YOLO.download_dataset("voc2007")

settings = YOLO.pretrained.v2_tiny_voc.load(minibatch_size=1) #run 1 image at a time
model = YOLO.v2_tiny.load(settings)
YOLO.loadWeights!(model, settings)

voc = YOLO.datasets.VOC.populate()
vocloaded = YOLO.load(voc, settings, indexes = [100]) #load image #100 (a single image)

#Run the model
res = model(vocloaded.imstack_mat);

#Convert the output into readable predictions
predictions = YOLO.postprocess(res, settings, conf_thresh = 0.3, iou_thresh = 0.3)
```

### Rendering results
To render results, first load `Makie` before `YOLO` (in a fresh julia instance):
```julia
using Makie, YOLO
## Repeat all above steps to load & run the model
scene = YOLO.renderResult(vocloaded.imstack_mat[:,:,:,1], predictions, settings, save_file = "test.png")
display(scene)
```


### Testing inference speed

#### Model + post-process
```julia
using BenchmarkTools
@benchmark begin
  res = model(vocloaded.imstack_mat);
  predictions = YOLO.postprocess(res, settings, conf_thresh = 0.3, iou_thresh = 0.3)
end
```
Results for model + postprocess on a modern macbook. CPU-only: ~10 FPS
```
BenchmarkTools.Trial:
  memory estimate:  124.39 MiB
  allocs estimate:  12953
  --------------
  minimum time:     97.784 ms (13.12% GC)
  median time:      115.836 ms (11.17% GC)
  mean time:        115.455 ms (13.33% GC)
  maximum time:     137.506 ms (7.29% GC)
  --------------
  samples:          44
  evals/sample:     1
```

#### Model only
```julia
using BenchmarkTools
@benchmark model(vocloaded.imstack_mat)
```
Results for model-only on a desktop with Gtx 1070 GPU: ~267 FPS
```
BenchmarkTools.Trial: 
  memory estimate:  56.92 KiB
  allocs estimate:  1442
  --------------
  minimum time:     375.116 μs (0.00% GC)
  median time:      3.740 ms (0.00% GC)
  mean time:        3.738 ms (2.65% GC)
  maximum time:     10.810 ms (0.00% GC)
  --------------
  samples:          1337
  evals/sample:     1
```





[discourse-tag-url]: https://discourse.julialang.org/tags/yolo

[travis-img]: https://travis-ci.com/ianshmean/YOLO.jl.svg?branch=master
[travis-url]: https://travis-ci.com/ianshmean/YOLO.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/ianshmean/YOLO.jl?svg=true
[appveyor-url]: https://ci.appveyor.com/project/ianshmean/YOLO-jl

[drone-img]: https://cloud.drone.io/api/badges/ianshmean/YOLO.jl/status.svg
[drone-url]: https://cloud.drone.io/ianshmean/YOLO.jl

[cirrus-img]: https://api.cirrus-ci.com/github/ianshmean/YOLO.jl.svg
[cirrus-url]: https://cirrus-ci.com/github/ianshmean/YOLO.jl

[codecov-img]: https://codecov.io/gh/ianshmean/YOLO.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/ianshmean/YOLO.jl

[coveralls-img]: https://coveralls.io/repos/github/ianshmean/YOLO.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/ianshmean/YOLO.jl?branch=master

[issues-url]: https://github.com/ianshmean/YOLO.jl/issues
