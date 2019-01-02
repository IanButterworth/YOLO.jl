# FluxYOLOv3.jl
# utils/datasets.jl

using Random, Glob

using Flux


mutable struct ImageFolder
    function ImageFolder(folder_path;img_size=416)
        files = sort(Glob.glob("*.*",folder_path))
        img_shape = (img_size, img_size)
    end
end
