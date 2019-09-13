#Author: Yavuz Faruk Bakman
#Date: 15/08/2019

#collects all labels and images' directories
function inputandlabelsdir(dirlab,dirinput)
    println("Collecting input and labels' directories")
    labels = []
    images = []
    for (root, dirs, files) in walkdir(mkpath(dirlab);)
        for file in files
            if occursin(".xml",file)
                tolabel = joinpath(root,file)
                jpgFile = string(file[1:length(file)-3], "jpg")
                toimage = joinpath(dirinput,jpgFile)
                push!(labels,tolabel)
                push!(images,toimage)
            end
        end
    end
    println("Collecting done")
    return images,labels
end

#collects input directories
function inputdir(inputdir)
    images  = []
    println("Collecting input directories")
    for (root, dirs, files) in walkdir(mkpath(inputdir);)
        for file in files
            toimage = joinpath(root,file)
            push!(images,toimage)
        end
    end
    return images
    println("Collecting done")
end

#prepares input and its' labels
function prepareinputlabels(inArr,labArr)
    in,imgs = prepareinput(inArr)
    lab = preparelabels(labArr)
    lab = arrangelabels(lab,416)
    return in,lab,imgs
end

prepInput(inRes,imgs,data) =(prepInput!(inRes,imgs,args) for args in data)

function prepInput!(inRes,imgs,args)
    im, img_size, img_originalsize, padding = loadprepareimage(args,(416,416))
    im_input = Array{Float32}(undef,416,416,3,1)
    im_input[:,:,:,1] = permutedims(collect(channelview(im)),[2,3,1])
    push!(inRes,im_input)
    push!(imgs,im)
end

function prepareinput(inArr)
    inRes = Array{Array{Float32,4},1}()
    imgs= []
    println("Pre-processing images")
    progress!(prepInput(inRes,imgs,inArr))
    println("Pre-processing done")
    return cat(inRes...,dims=4),imgs
end

preplabels(labArr,labRes) =(preplabels!(args,labRes) for args in labArr)

#prepares labels
function preplabels!(args,labRes)
    toPush = []
    xdoc = parse_file(args)
    xroot = root(xdoc)
    ces = get_elements_by_tagname(xroot, "size")
    width = parse(Int32,content(find_element(ces[1], "width")))
    height = parse(Int32,content(find_element(ces[1], "height")))
    push!(toPush,width)
    push!(toPush,height)
    ces = get_elements_by_tagname(xroot, "object")
    for i in 1:length(ces)
        obj = []
        name= content(find_element(ces[i], "name"))
        difficult = content(find_element(ces[i], "difficult"))
        if difficult == "0"
            totaldic[name] = totaldic[name] + 1
            #get xmin xmax ymin ymax
            xmin = parse(Int32,content(find_element(find_element(ces[i], "bndbox"),"xmin")))
            xmax = parse(Int32,content(find_element(find_element(ces[i], "bndbox"),"xmax")))
            ymin = parse(Int32,content(find_element(find_element(ces[i], "bndbox"),"ymin")))
            ymax = parse(Int32,content(find_element(find_element(ces[i], "bndbox"),"ymax")))
            push!(obj,xmin)
            push!(obj,ymin)
            push!(obj,xmax-xmin)
            push!(obj,ymax-ymin)
            push!(obj,name)
            push!(toPush,obj)
        end

    end
    push!(labRes,toPush)
end

function preparelabels(labArr)
    labRes = []
    println("Preparing labels...")
    progress!(preplabels(labArr,labRes))
    println("Labels are done")
    return labRes
end

arrlabels(lab,size) =(arrlabels!(args,size) for args in lab)

function arrlabels!(args,size)
    w = args[1]
    h = args[2]
    for k in 3:length(args)
        m = max(w,h)
        rate = size/m
        if w >= h
            pad = floor((size - h*rate)/2)
            args[k][1] = floor(args[k][1]*rate)
            args[k][2] = floor(args[k][2]*rate) + pad
            args[k][3] = floor(args[k][3]*rate)
            args[k][4] = floor(args[k][4]*rate)
        else
            pad = floor((size - w*rate)/2)
            args[k][1] = floor(args[k][1]*rate) + pad
            args[k][2] = floor(args[k][2]*rate)
            args[k][3] = floor(args[k][3]*rate)
            args[k][4] = floor(args[k][4]*rate)
        end
    end
end
# return all tupples as(ImageWidth, ImageHeight,[x,y,objectWidth,objectHeight],ImageHeight,[x,y,objectWidth,objectHeight]..)
function arrangelabels(lab,size)
    println("Arranging labels...")
    progress!(arrlabels(lab,size))
    println("Labels are arranged")
    return lab
end


#prepares an image as given shapes
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
    imgrp = padarray(imgr, Fill(ColorTypes.RGB(0.0,0.0,0.0),(vpad_top,hpad_left),(vpad_bottom,hpad_right)))
    return imgrp, img_size, img_originalsize, padding
end
