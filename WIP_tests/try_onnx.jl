
# First download and unzip from https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3

using Flux, ONNX                             # Import the required packages.
ONNX.load_model("yolov3.onnx")                # If you are in some other directory, specify the entire path.
# This creates two files: model.jl and weights.bson.

weights = ONNX.load_weights("weights.bson")  # Read the weights from the binary serialized file.
model = include("model.jl")                  # Loads the model from the model.jl file.
