using YOLO, BenchmarkTools

#First time only (downloads 5011 images & labels!)
#YOLO.download_dataset("voc2007")

settings = YOLO.pretrained.v2_tiny_voc.load(minibatch_size=1) #Process one image at a time
model = YOLO.v2_tiny.load(settings)
YOLO.loadWeights!(model, settings)

voc = YOLO.datasets.VOC.populate()
vocloaded = YOLO.load(voc, settings, indexes = [rand(1:5011)]) #Load a single random image

bt = @benchmark begin
  res = model(vocloaded.imstack_mat);
  predictions = YOLO.postprocess(res, settings, conf_thresh = 0.3, iou_thresh = 0.3)
end

display(bt)
