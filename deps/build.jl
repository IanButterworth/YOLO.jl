
@info "Downloading weights files"
weightsdir = joinpath(@__DIR__,"..","data","pretrained")
v2_tiny_dir = joinpath(weightsdir,"v2_tiny","voc")
download("https://pjreddie.com/media/files/yolov2-tiny-voc.weights",
                        joinpath(v2_tiny_dir,"v2_tiny_voc.weights"))

@info "Downloading datasets"
datasetsdir = joinpath(@__DIR__,"..","data","datasets")
!isdir(datasetsdir) && mkdir(datasetsdir)

voc_dir = joinpath(datasetsdir,"voc")
!isdir(voc_dir) && mkdir(voc_dir)
tmploc = download("https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar")
run(`tar xf $tmploc -C $(voc_dir)`)

@info "Build complete"
