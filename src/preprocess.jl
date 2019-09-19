"""
    resizekern(source_size::Tuple{Int,Int}, dest_size::Tuple{Int,Int})

Create an image resize blur kernel, for use before reducing image size to avoid aliasing.
"""
function resizekern(source_size::Tuple{Int,Int}, dest_size::Tuple{Int,Int})
    # Blur before resizing to prevent aliasing (kernel size dependent on both source and target image size)
    σ = map((o, n) -> 0.75 * o / n, source_size, dest_size)
    if first(σ) < 1
        return ImageFiltering.KernelFactors.gaussian(σ)
    else
        return ImageFiltering.KernelFactors.IIRGaussian(σ)
    end
end

"""
    sizethatfits(original_size::Tuple{Int,Int},target_shape::Tuple{Int,Int})

Takes an original image size, and fits it to a target shape for image resizing. Maintains aspect ratio.
"""
function sizethatfits(
    original_size::Tuple{Int,Int},
    target_shape::Tuple{Int,Int},
)
    if original_size[1] > original_size[2]
        target_img_size = (
            target_shape[1],
            floor(Int, target_shape[2] * (original_size[2] / original_size[1])),
        )
    else
        target_img_size = (
            floor(Int, target_shape[1] * (original_size[1] / original_size[2])),
            target_shape[2],
        )
    end
    return target_img_size
end

"""
    loadResizePadImageToFit(img_path::String, sets::Settings)

Load image and reesize it to fit inside the target image shap, while maintaining
aspect ratio and preventing aliasing.
"""
function loadResizePadImageToFit(
    img_path::String,
    sets::Settings
)
    img = FileIO.load(img_path)
    img_size = size(img)
    target_img_size = sizethatfits(img_size, sets.image_shape)
    kern = resizekern(img_size, target_img_size)
    return resizePadImageToFit(img, target_img_size, sets, kern)
end

"""
    resizePadImageToFit(img_path::String, sets::Settings, kern::Tuple{ImageFiltering.KernelFactors.ReshapedOneD,ImageFiltering.KernelFactors.ReshapedOneD})

Loads and prepares (resizes + pads) an image to fit within a given shape.
Returns the image and the padding.
"""
function resizePadImageToFit(
    img::Array{T},
    target_img_size::Tuple{Int,Int},
    settings::Settings,
    kern::Tuple{
        ImageFiltering.KernelFactors.ReshapedOneD,
        ImageFiltering.KernelFactors.ReshapedOneD,
    },
) where {T<:ColorTypes.Color}

    imgr = ImageTransformations.imresize(
        ImageFiltering.imfilter(img, kern, NA()),
        target_img_size,
    )

    # Determine top and left padding
    vpad_top = floor(Int, (settings.image_shape[1] - target_img_size[1]) / 2)
    hpad_left = floor(Int, (settings.image_shape[2] - target_img_size[2]) / 2)

    # Determine bottom and right padding accounting for rounding of top and left (to ensure accuate result image size if source has odd dimensions)
    vpad_bottom = settings.image_shape[1] - (vpad_top + target_img_size[1])
    hpad_right = settings.image_shape[2] - (hpad_left + target_img_size[2])

    padding = [hpad_left, vpad_top, hpad_right, vpad_bottom]

    # Pad image
    return padarray(
            imgr,
            Fill(
                zero(eltype(img)),
                (vpad_top, hpad_left),
                (vpad_bottom, hpad_right),
            ),
        ),
        padding
end

"""
    load(ds::LabelledImageDataset, settings::Settings; limitfirst::Int = -1)

Load images from a populated `LabelledImageDataset` into memory.
"""
function load(
    ds::LabelledImageDataset,
    settings::Settings;
    indexes::Union{Array{Int}} = [],
)
    if length(indexes) == 1
        numimages = 1
    elseif length(indexes) > 0 && length(indexes) < length(ds.image_paths)
        numimages = length(indexes)
        @info "Loading $(numimages) images from $(ds.name) dataset into memory"
    else
        numimages = length(ds.image_paths)
        indexes = 1:numimages
        @info "Loading all ($(numimages)) images from $(ds.name) dataset into memory"
    end
    kern = resizekern(ds.image_size_lims, settings.image_shape)

    firstimg, padding = loadResizePadImageToFit(
        ds.image_paths[1],
        settings,
    )
    imgsize = size(firstimg)
    lds = LoadedDataset(
            imstack_mat = xtype(undef,imgsize[1],imgsize[2],settings.image_channels,numimages),
            paddings = Vector{Vector{Int}}(undef, 0),
            labels = ds.labels[indexes],
        )

    j = 1
    ProgressMeter.@showprogress 2 "Loading images..." for i in indexes
        img = FileIO.load(ds.image_paths[i])
        img_size = size(img)
        target_img_size = sizethatfits(img_size, settings.image_shape)
        img_resized, padding = resizePadImageToFit(
            img,
            target_img_size,
            settings,
            kern,
        )
        lds.imstack_mat[:, :, :, j] = collect(permutedims(
            channelview(img_resized)[1:settings.image_channels, :, :],
            [2, 3, 1],
        ))
        push!(lds.paddings, padding)
        j += 1
    end
    return lds
end
