# YOLO.jl

Currently only supports loading YOLOv2-tiny and the VOC pretrained model (pretrained by Darknet).

The majority of this is made possible by Yavuz Bakman's great work in https://github.com/Ybakman/YoloV2

**Docs**
[![Join the julia slack](https://img.shields.io/badge/chat-slack%23machine%2Dlearning-yellow.svg)](https://slackinvite.julialang.org)

| **Platform**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| Linux & MacOS x86 | [![][travis-img]][travis-url] |
| Windows 32/64-bit | [![][appveyor-img]][appveyor-url] |
| Linux ARM 32/64-bit | [![][drone-img]][drone-url] |
| FreeBSD x86 | [![][cirrus-img]][cirrus-url] |
|  | [![Codecoverage Status][codecov-img]][codecov-url]<br>[![Coveralls Status](coveralls-badge)](coveralls-url) * |


## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add VideoIO
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("VideoIO")
```

## Example Usage (WIP)
```julia
using YOLO

#First time only (downloads 5011 images & labels!)
YOLO.download_dataset("voc2007")

settings = YOLO.pretrained.v2_tiny_voc.load(minibatch_size=1)
model = YOLO.v2_tiny.load(settings)
YOLO.loadWeights!(model, settings)

voc = YOLO.datasets.VOC.populate()
vocloaded = YOLO.load(voc, settings, indexes = [100])

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
```julia
using BenchmarkTools
@btime begin
  res = model(vocloaded.imstack_mat);
  predictions = YOLO.postprocess(res, settings, conf_thresh = 0.3, iou_thresh = 0.3)
end
```
Results on a modern macbook. CPU-only:
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
i.e. ~10 FPS





[discourse-tag-url]: https://discourse.julialang.org/tags/yolo

[travis-img]: https://travis-ci.com/ianshmean/YOLO.jl.svg?branch=master
[travis-url]: https://travis-ci.com/ianshmean/YOLO.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/c1nc5aavymq76xun?svg=true
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
