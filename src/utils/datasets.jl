# FluxYOLOv3.jl
# utils/datasets.jl

using Random, Glob, FileIO, DelimitedFiles, OffsetArrays
using Images, ImageDraw, ImageFiltering, ImageTransformations, Colors

using FreeTypeAbstraction

using Flux

# load a font
face = newface(string(@__DIR__,"/droid-sans-mono/DroidSansMono.ttf"))


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
function loadimagefolder(folder_path::String,img_size::Int)
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

    img, img_size, img_originalsize, padding =  loadprepareimage(img_path,img_shape)

    return img_path, img
end

function loadprepareimage(img_path::String,img_shape::Tuple{Int,Int})
    #Extract image
    img = load(img_path)
    img_originalsize = size(img)

    if img_originalsize[1] > img_originalsize[2]
        img_size = (img_shape[1],floor(Int,img_shape[2]*(img_originalsize[2]/img_originalsize[1])))
    else
        img_size = (floor(Int,img_shape[1]*(img_originalsize[1]/img_originalsize[2])),img_shape[2])
    end

    # Resize after blurring to prevent aliasing
    σ = map((o,n)->0.75*o/n, size(img), img_size)
    kern = KernelFactors.gaussian(σ)   # from ImageFiltering
    imgr = imresize(imfilter(img, kern, NA()), img_size)

    # Determine top and left padding
    vpad_top = floor(Int,(img_shape[1]-img_size[1])/2)
    hpad_left = floor(Int,(img_shape[2]-img_size[2])/2)

    # Determine bottom and right padding accounting for rounding of top and left (to ensure accuate result image size if source has odd dimensions)
    vpad_bottom = img_shape[1] - (vpad_top + img_size[1])
    hpad_right = img_shape[2] - (hpad_left + img_size[2])

    padding = [hpad_left,vpad_top,hpad_right,vpad_bottom]

    # Pad image
    imgrp = padarray(imgr, Fill(ColorTypes.RGB(0.5,0.5,0.5),(vpad_top,hpad_left),(vpad_bottom,hpad_right)))

    return imgrp, img_size, img_originalsize, padding
end


# ListDataset
mutable struct ListDataset
    img_files::Array{String,1}
    label_files::Array{String,1}
    img_shape::Tuple{Int,Int}
    max_objects::Int
end
function loadlistdataset(list_path::String,img_size::Int)
    img_files = readlines(list_path)
    label_files = map(s->foldl(replace, [("/images/" => "/labels/"),(".png" => ".txt"),(".jpg" => ".txt")], init=s),img_files)
    img_shape = (img_size, img_size)
    max_objects = 50
    return ListDataset(img_files,label_files,img_shape,max_objects)
end
function length(ListDataset::ListDataset)
    return length(ListDataset.img_files)
end
function getitem(ListDataset::ListDataset,i::Int)
    img_path = ListDataset.img_files[i]
    img_shape = ListDataset.img_shape

    # Image: Load resize and pad
    img, img_size, img_originalsize, img_padding =  loadprepareimage(img_path,img_shape)

    h, w = img_originalsize
    padded_h, padded_w = img_shape

    # Label:
    label_path = strip(ListDataset.label_files[i])

    # Fill matrix
    filled_labels = zeros(Float64,ListDataset.max_objects,5)

    if isfile(label_path)
        labels = readdlm(label_path)
        imax = minimum([ListDataset.max_objects,size(labels,1)])
        filled_labels[1:imax,:] = labels
    end

    return LabelledImage(img, img_path, img_size, img_padding, filled_labels)
end

# LabelledImage
mutable struct LabelledImage
    img::OffsetArray{RGB{Float64}}
    img_path::String
    img_size::Tuple{Int,Int}
    img_padding::Array{Int64}
    filled_labels::Array{Float64}
end

"""
Base.show(io::IO,x::LabelledImage)

Show labelled image with boxes overlaid from labels
"""
function Base.show(io::IO,x::LabelledImage)
    imview = x.img
    h,w = x.img_size
    for i = 1:size(x.filled_labels,1)
        label = x.filled_labels[i,:]
        if sum(label) !== 0.0
            lab = string(round(Int,label[1]))
            #lab="dsafad"
            drawsquare(imview,label[2:5],w,h,lab)
        end
    end
    display(imview)
end

function drawsquare(im,bbox::Array{Float64},w,h,label)
    pbbox = getpixelbbox(bbox,w,h)
    corners = getpixelcorners(bbox,w,h)
    draw!(im, LineSegment(Point(corners[1]), Point(corners[2])))
    draw!(im, LineSegment(Point(corners[2]), Point(corners[3])))
    draw!(im, LineSegment(Point(corners[3]), Point(corners[4])))
    draw!(im, LineSegment(Point(corners[4]), Point(corners[1])))
    try
        FreeTypeAbstraction.renderstring!(im, string(label), face, (14,14), corners[4][2], corners[4][1]+1, halign=:hleft,valign=:vtop,bcolor=RGB{Float64}(1.0,1.0,1.0),fcolor=RGB{Float64}(0,0,0)) #use `nothing` to make bcolor transparent
    catch e
        warn("Couldn't draw bbox label due to label falling outside of image bounds")
    end
end
function getpixelbbox(bbox::Array{Float64},w,h)
    return [round(Int,1+(bbox[1]*(w-1))),round(Int,1+(bbox[2]*(h-1))),trunc(Int,(bbox[3]*w)),trunc(Int,(bbox[4]*h))]
end
function getpixelcorners(bbox::Array{Float64},w,h)
    pbbox = getpixelbbox(bbox,w,h)
    topleft =       (clamp(pbbox[1]-trunc(Int,pbbox[3]/2),1,w),clamp(pbbox[2]+trunc(Int,pbbox[4]/2),1,h))
    topright =      (clamp(pbbox[1]+trunc(Int,pbbox[3]/2),1,w),clamp(pbbox[2]+trunc(Int,pbbox[4]/2),1,h))
    bottomright =   (clamp(pbbox[1]+trunc(Int,pbbox[3]/2),1,w),clamp(pbbox[2]-trunc(Int,pbbox[4]/2),1,h))
    bottomleft =    (clamp(pbbox[1]-trunc(Int,pbbox[3]/2),1,w),clamp(pbbox[2]-trunc(Int,pbbox[4]/2),1,h))
    return [topleft,topright,bottomright,bottomleft]
end
