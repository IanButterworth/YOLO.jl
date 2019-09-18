module datasets
import ..datasets_dir
include(joinpath(datasets_dir, "voc", "voc.jl"))
end #module

"""
    download_dataset(;name="all")

Download supported datasets. If folder exists, deletes first.
"""
function download_dataset(name::String = "all")
    !isdir(datasets_dir) && mkdir(datasets_dir)
    if any(name .== ["voc", "VOC", "all"])
        voc_dir = joinpath(datasets_dir, "voc")
        voc_root = joinpath(voc_dir, "VOCdevkit")
        isdir(voc_root) && rm(voc_root, force = true, recursive = true)
        @info "Downloading dataset..."
        tmploc = download("https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar")
        @info "Extracting..."
        run(`tar xf $tmploc -C $voc_dir`)
        @info "Completed"
    end
end
