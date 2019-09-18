# YOLO.jl

Currently only supports loading YOLOv2-tiny and the VOC pretrained model (pretrained by Darknet).

The majority of this is made possible by Yavuz Bakman's great work in https://github.com/Ybakman/YoloV2


## Example Usage (WIP)
```julia
using YOLO

#First time only (downloads 5011 images & labels!)
YOLO.download_dataset("voc")

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
## Repeat above steps
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
