# CREDIT: https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh

using ZipFile, DelimitedFiles

# Set dir to location of this script
cd(@__DIR__)

# Clone COCO API
if !isdir("coco")
    run(`git clone https://github.com/pdollar/coco`)
end
cd("coco")

if !isdir("images")
    run(`mkdir images`)
end
cd("images")

# Download Images
println("Downloading image archives (~20 GB)...")
if !isfile("train2014.zip")
    download("https://pjreddie.com/media/files/train2014.zip","train2014.zip")
end
if !isfile("val2014.zip")
    download("https://pjreddie.com/media/files/val2014.zip","val2014.zip")
end

# Unzip
println("Unzipping image archives...")
run(`unzip -q "train2014.zip"`)
run(`unzip -q "val2014.zip"`)

cd("..")

# Download COCO Metadata
println("Downloading COCO Metadata...")
download("https://pjreddie.com/media/files/instances_train-val2014.zip","instances_train-val2014.zip")
download("https://pjreddie.com/media/files/coco/5k.part","5k.part")
download("https://pjreddie.com/media/files/coco/trainvalno5k.part","trainvalno5k.part")
download("https://pjreddie.com/media/files/coco/labels.tgz","labels.tgz")
run(`tar xzf labels.tgz`)
run(`unzip -q "instances_train-val2014.zip"`)

PWD = pwd()
# Set Up Image Lists
writedlm("5k.txt", map(x->string(PWD,x),readlines("5k.part")))
writedlm("trainvalno5k.txt", map(x->string(PWD,x),readlines("trainvalno5k.part")))
