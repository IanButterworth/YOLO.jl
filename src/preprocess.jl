#Author: Yavuz Faruk Bakman
#Date: 15/08/2019


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
