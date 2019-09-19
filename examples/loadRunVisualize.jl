using Makie, YOLO, BenchmarkTools

#First time only (downloads 5011 images & labels!)
YOLO.download_dataset("voc2007")

settings = YOLO.pretrained.v2_tiny_voc.load(minibatch_size=1) #Process one image at a time
model = YOLO.v2_tiny.load(settings)
YOLO.loadWeights!(model, settings)

voc = YOLO.datasets.VOC.populate()
vocloaded = YOLO.load(voc, settings, indexes = [rand(1:5011)]) #Load a single random image

#Run the model
res = model(vocloaded.imstack_mat);

#Convert the output into readable predictions
predictions = YOLO.postprocess(res, settings, conf_thresh = 0.3, iou_thresh = 0.3)

scene = YOLO.renderResult(vocloaded.imstack_mat[:,:,:,1], predictions, settings, save_file = "test.png")
display(scene)

@btime begin
  res = model(vocloaded.imstack_mat);
  predictions = YOLO.postprocess(res, settings, conf_thresh = 0.3, iou_thresh = 0.3)
end
