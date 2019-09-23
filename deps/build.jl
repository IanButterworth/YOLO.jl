
weightsdir = joinpath(@__DIR__, "..", "data", "pretrained")

@info "Downloading weights files"

download(
    "https://github.com/ianshmean/YOLO.jl/releases/download/v2-tiny-VOC/v2_tiny_voc.weights",
    joinpath(weightsdir, "v2_tiny", "voc2007", "v2_tiny_voc.weights"),
)

download(
    "https://pjreddie.com/media/files/yolov2-voc.weights",
    joinpath(weightsdir, "v2", "voc2007", "v2_voc.weights"),
)

@info "Build complete"
