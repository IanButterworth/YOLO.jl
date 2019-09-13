"""
    download_dataset(;name="all")

Download supported datasets. If folder exists, deletes first.
"""
function download_dataset(name::String = "all")
    !isdir(datasets_dir) && mkdir(datasets_dir)
    if name == "voc" || name == "all"
        voc_dir = joinpath(datasets_dir, "voc")
        isdir(voc_dir) && rm(voc_dir, force = true, recursive = true)
        mkdir(voc_dir)
        tmploc = download("https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar")
        run(`tar xf $tmploc -C $(voc_dir)`)
    end
end
