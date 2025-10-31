module v2_tiny
#Tiny Yolo V2-tiny model configuration
include("../../YoloLoss.jl")

using Knet: Knet, progress, progress!, gpu, KnetArray, relu, minibatch, conv4, pool, softmax

import ..sigmoid, ..Settings

mutable struct v2TinyChain
    layers
    v2TinyChain(layers...) = new(layers)
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
    grid_x::Int
    grid_y::Int
end
struct Pool # Define pool layer
    size
    stride
    pad
end

YoloPad(w1::Int, w2::Int, cx::Int, cy::Int, grid_x::Int, grid_y::Int) =
    YoloPad(zeros(Float32, w1, w2, cx, cy), grid_x, grid_y)#Constructor for Yolopad

Conv(w1::Int, w2::Int, cx::Int, cy::Int, st, pd, f) =
    Conv(randn(Float32, w1, w2, cx, cy), randn(Float32, 1, 1, cy, 1), st, pd, f)#Constructor for convolutional layer

#Assymetric padding function
function (y::YoloPad)(x)
    x = reshape(x, y.grid_x + 1, y.grid_y + 1, 1, :)
    return reshape(conv4(y.w, x; stride = 1), y.grid_x, y.grid_y, 512, :)
end

(p::Pool)(x) = pool(x; window = p.size, stride = p.stride, padding = p.pad) #pool function

(c::v2TinyChain)(x) = (for l in c.layers #chain function
    x = l(x)
end; x)


(c::v2TinyChain)(x, truth) = yololoss(truth,c(x))#Array{Float32}(m(x))

(c::Conv)(x) =
    c.f.(conv4(c.w, x; stride = c.stride, padding = c.padding) .+ c.b) #convolutional layer function

#leaky function
function leaky(x)
    return max(Float32(0.1) * x, x)
end

"""
    load(sets::Settings)

Loads the YOLOv2_tiny model.

Note on Asymmetric Padding (see https://github.com/Ybakman/YoloV2#asymmetric-padding)
An alternative solution for asymmetric padding on pooling. We solved the asymmetric padding process in the following 4 steps: Let's assume we have a matrix as (d1,d2,depth,minibatch_size)

1- Apply symmetric padding = 1 with the pooling function

2- reshape it into => (d1,d2,1,minibatch_size*depth)

3- apply the following convolutional layer => 1 0  with stride = 1, padding = 0
                                              0 0

4- Lastly, reshape again into => (d1,d2,depth,minibatch_size)

Now, we made pooling with asymmetric padding on the right and bottom side. If you want to
make it on left and top side, apply this conv layer =>  0 0
                                                        0 1

The last 3 steps are implemented in YoloPad(x) function
The symmetric padding part is implemented in the pooling function just before Yolopad(x)
"""
function load(sets::Settings)
return v2TinyChain(
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
    YoloPad(2, 2, 1, 1, sets.grid_x, sets.grid_y),
    Conv(3, 3, 512, 1024, 1, 1, leaky),
    Conv(3, 3, 1024, 1024, 1, 1, leaky),
    Conv(1, 1, 1024, sets.cell_bboxes * (5 + sets.num_classes), 1, 0, identity),
)
end

include(joinpath(@__DIR__, "loadweights.jl"))

end #module
