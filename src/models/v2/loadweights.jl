####
# NOT FUNCTIONAL
#TODO: Convert this to v2 model - This will not work for v2
####


import ..GPU


import ..YOLO: loadWeights!

export loadWeights!

"""
    flipkernel(x)

Flips given kernel
"""
flipkernel(x) = x[end:-1:1, end:-1:1, :, :]

"""
    updateconv!(c, gamma, mean, variance)

Applies batch-normalization to given convolutional layer
"""
function updateconv!(c, gamma, mean, variance)
    gamma4 = reshape(gamma, 1, 1, 1, :)
    variance4 = reshape(variance, 1, 1, 1, :)
    gamma3 = reshape(gamma, 1, 1, :, 1)
    mean3 = reshape(mean, 1, 1, :, 1)
    variance3 = reshape(variance, 1, 1, :, 1)
    #update c.w
    a = convert(Float32, 0.001)
    c.w = (c.w .* gamma4) ./ sqrt.(variance4 .+ a)
    #update c.b
    c.b = c.b .- (gamma3 .* mean3 ./ sqrt.(variance3 .+ a))
    return c.w, c.b
end

"""
    loadWeights!(model::YOLO.v2_tiny.v2Chain, weights_filepath::String)

Loads layers' weights from given weights file path.
"""
function loadWeights!(model::v2Chain, settings::Settings)
    outlength = settings.cell_bboxes * (5 + settings.num_classes)

    open(settings.weights_filepath, "r") do io
        readconstants!(io)
        #First Conv layer
        loadconv!(model.layers[1], io, 3, 3, 3, 16)
        #Second Conv layer
        loadconv!(model.layers[3], io, 3, 3, 16, 32)
        #Third Conv layer
        loadconv!(model.layers[5], io, 3, 3, 32, 64)
        #4th Conv layer
        loadconv!(model.layers[7], io, 3, 3, 64, 128)
        #5th Conv layer
        loadconv!(model.layers[9], io, 3, 3, 128, 256)
        #6th Conv layer
        loadconv!(model.layers[11], io, 3, 3, 256, 512)
        #YoloPad
        model.layers[13].w[1, 1, 1, 1] = 1
        #7th Conv layer
        loadconv!(model.layers[14], io, 3, 3, 512, 1024)
        #8th Conv layer
        loadconv!(model.layers[15], io, 3, 3, 1024, 1024)
        #last layer
        read!(io, model.layers[16].b)
        toRead = Array{Float32}(undef, 1024 * outlength)
        read!(io, toRead)
        toRead = reshape(toRead, 1, 1, 1024, outlength)
        model.layers[16].w = permutedims(toRead, [2, 1, 3, 4])
        model.layers[16].w = flipkernel(model.layers[16].w)

        if GPU >= 0
            model.layers[16].w = KnetArray(model.layers[16].w)
            model.layers[16].b = KnetArray(model.layers[16].b)
            model.layers[13].w = KnetArray(model.layers[13].w)
        end
    end
end

"""
    loadconv!(c, io, d1, d2, d3, d4)

Loads the io to given convolutional layer and updates it by batch-normalization
"""
function loadconv!(c, io::IOStream, d1, d2, d3, d4)
    read!(io, c.b)
    gamma = Array{Float32}(undef, d4)
    mean = Array{Float32}(undef, d4)
    variance = Array{Float32}(undef, d4)
    read!(io, gamma)
    read!(io, mean)
    read!(io, variance)
    toRead = Array{Float32}(undef, d4 * d3 * d2 * d1)
    read!(io, toRead)
    toRead = reshape(toRead, d1, d2, d3, d4)
    c.w = permutedims(toRead, [2, 1, 3, 4])
    c.w, c.b = updateconv!(c, gamma, mean, variance)
    c.w = flipkernel(c.w)
    if GPU >= 0
        c.w = KnetArray(c.w)
        c.b = KnetArray(c.b)
    end
end

"""
    readconstants!(io)

Read constant and unnecessary numbers from the io stream
"""
function readconstants!(io::IOStream)
    major = read(io, Int32)
    minor = read(io, Int32)
    revision = read(io, Int32)
    iseen = read(io, Int32)
end
