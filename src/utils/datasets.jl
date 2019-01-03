# FluxYOLOv3.jl
# utils/datasets.jl

using Random, Glob, FileIO, ImageTransformations, ImageFiltering, Colors

using Flux

# ImageFolder
mutable struct ImageFolder
    folder_path::String
    img_shape::Tuple{Int,Int}
    files::Array{String,1}
end
"""
LoadImageFolder(folder_path::String,img_size::Int)
Populate ImageFolder type with contents of folder
"""
function LoadImageFolder(folder_path::String,img_size::Int)
    img_shape = (img_size, img_size)
    files = sort(Glob.glob("*.*",folder_path))
    return ImageFolder(folder_path,img_shape,files)
end
function length(ImageFolder::ImageFolder)
    return length(ImageFolder.files)
end
function getitem(ImageFolder::ImageFolder,i::Int64)
    img_shape = ImageFolder.img_shape
    img_path = ImageFolder.files[i]
    #Extract image
    img = load(img_path)
    img_size = size(img)

    if img_size[1] > img_size[2]
        sz = (img_shape[1],floor(Int,img_shape[2]*(img_size[2]/img_size[1])))
    else
        sz = (floor(Int,img_shape[1]*(img_size[1]/img_size[2])),img_shape[2])
    end

    # Resize after blurring to prevent aliasing
    σ = map((o,n)->0.75*o/n, size(img), sz)
    kern = KernelFactors.gaussian(σ)   # from ImageFiltering
    imgr = imresize(imfilter(img, kern, NA()), sz)

    # Determine top and left padding
    vpad_top = floor(Int,(img_shape[1]-sz[1])/2)
    hpad_left = floor(Int,(img_shape[2]-sz[2])/2)

    # Determine bottom and right padding accounting for rounding of top and left (to ensure accuate result image size if source has odd dimensions)
    vpad_bottom = img_shape[1] - (vpad_top + sz[1])
    hpad_right = img_shape[2] - (hpad_left + sz[2])

    # Pad image
    imgrp = padarray(imgr, Fill(ColorTypes.RGB(0.5,0.5,0.5),(vpad_top,hpad_left),(vpad_bottom,hpad_right)))

    return img_path, imgrp
end

# ListDataset
mutable struct ListDataset
    img_files::Array{String,1}
    label_files::Array{String,1}
    img_shape::Tuple{Int,Int}
    max_objects::Int
end
function LoadListDataset(list_path::String,img_size::Int)
    img_files = readlines(list_path)
    label_files = map(s->foldl(replace, [("/images/" => "/labels/"),(".png" => ".txt"),(".jpg" => ".txt")], init=s),img_files)
    img_shape = (img_size, img_size)
    max_objects = 50
    return ListDataset(img_files,label_files,img_shape,max_objects)
end
function length(ListDataset::ListDataset)
    return length(ListDataset.img_files)
end
