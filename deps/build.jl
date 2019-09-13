
@info "Downloading weights files"
weightsdir = joinpath(@__DIR__,"..","data","pretrained")
v2_tiny_dir = joinpath(weightsdir,"v2_tiny","voc")
download("https://pjreddie.com/media/files/yolov2-tiny-voc.weights",
                        joinpath(v2_tiny_dir,"v2_tiny_voc.weights"))

@info "Build complete"
