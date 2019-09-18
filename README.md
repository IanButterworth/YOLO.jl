# YOLO.jl - Work in progress!

Currently only supports loading YOLOv2-tiny and the VOC pretrained (by Darknet) model.

Loading of pretrainedMade possible by Yavuz Bakman's work in https://github.com/Ybakman/YoloV2


### Example (WIP)
```julia
using YOLO

#First time only (downloads 5011 images & labels!)
YOLO.download_dataset("voc")

settings = YOLO.pretrained.v2_tiny_voc.load(minibatch_size=1)
model = YOLO.v2_tiny.load(settings)
YOLO.loadWeights!(model, settings)

voc = YOLO.datasets.VOC.populate()
vocloaded = YOLO.load(voc, settings, indexes = [100])
res = model(vocloaded.imstack_mat);

predictions = YOLO.postprocess(res, settings, conf_thresh = 0.3, iou_thresh = 0.3)
scene = YOLO.renderResult(vocloaded.imstack_mat[:,:,:,1], predictions, settings, save_file = "test.png")
display(scene)
```


## Inference speed (on a modern macbook CPU)
```julia
using BenchmarkTools
@btime begin
  res = model(vocloaded.imstack_mat);
  predictions = YOLO.postprocess(res, settings, conf_thresh = 0.3, iou_thresh = 0.3)
end
```

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
