module datasets
import ..datasets_dir
include(joinpath(datasets_dir, "voc", "voc.jl"))
end #module

using BinaryProvider
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
        tmploc = download("https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar")
        BinaryProvider.unpack(tmploc, voc_dir)
        #run(`tar xf $tmploc -C $voc_dir`)
    end
end
