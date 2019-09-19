
weightsdir = joinpath(@__DIR__, "..", "data", "pretrained")

@info "Downloading weights files"

v2_tiny_dir = joinpath(weightsdir, "v2_tiny", "voc2007")
download(
    "https://github.com/ianshmean/YOLO.jl/releases/download/v2-tiny-VOC/v2_tiny_voc.weights",
    joinpath(v2_tiny_dir, "v2_tiny_voc.weights"),
)

@info "Build complete"
