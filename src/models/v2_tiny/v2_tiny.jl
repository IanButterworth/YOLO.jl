module v2_tiny
#Tiny Yolo V2-tiny model configuration

using Knet: Knet, progress, progress!, gpu, KnetArray, relu, minibatch, conv4, pool, softmax

import ..sigmoid, ..Settings

mutable struct Chain
    layers
    Chain(layers...) = new(layers)
end

mutable struct Conv #Define convolutional layer
    w
    b
    stride
    padding
    f
end
mutable struct YoloPad #Define Yolo padding layer (assymetric padding).
    w
    minibatch_size::Int
end
struct Pool # Define pool layer
    size
    stride
    pad
end

YoloPad(w1::Int, w2::Int, cx::Int, cy::Int, minibatch_size::Int) =
    YoloPad(zeros(Float32, w1, w2, cx, cy), minibatch_size)#Constructor for Yolopad

Conv(w1::Int, w2::Int, cx::Int, cy::Int, st, pd, f) =
    Conv(randn(Float32, w1, w2, cx, cy), randn(Float32, 1, 1, cy, 1), st, pd, f)#Constructor for convolutional layer

#Assymetric padding function
function (y::YoloPad)(x)
    x = reshape(x, 14, 14, 1, 512 * y.minibatch_size)
    return reshape(conv4(y.w, x; stride = 1), 13, 13, 512, y.minibatch_size)
end

(p::Pool)(x) = pool(x; window = p.size, stride = p.stride, padding = p.pad) #pool function

(c::Chain)(x) = (for l in c.layers #chain function
    x = l(x)
end; x)

(c::Conv)(x) =
    c.f.(conv4(c.w, x; stride = c.stride, padding = c.padding) .+ c.b) #convolutional layer function

#leaky function
function leaky(x)
    return max(Float32(0.1) * x, x)
end

function load(settings::Settings)
return Chain(
    Conv(3, 3, 3, 16, 1, 1, leaky),
    Pool(2, 2, 0),
    Conv(3, 3, 16, 32, 1, 1, leaky),
    Pool(2, 2, 0),
    Conv(3, 3, 32, 64, 1, 1, leaky),
    Pool(2, 2, 0),
    Conv(3, 3, 64, 128, 1, 1, leaky),
    Pool(2, 2, 0),
    Conv(3, 3, 128, 256, 1, 1, leaky),
    Pool(2, 2, 0),
    Conv(3, 3, 256, 512, 1, 1, leaky),
    Pool(2, 1, 1),
    YoloPad(2, 2, 1, 1, settings.minibatch_size),
    Conv(3, 3, 512, 1024, 1, 1, leaky),
    Conv(3, 3, 1024, 1024, 1, 1, leaky),
    Conv(1, 1, 1024, 125, 1, 0, identity),
)
end

include(joinpath(@__DIR__, "loadweights.jl"))

end #module
