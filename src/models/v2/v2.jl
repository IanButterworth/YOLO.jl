module v2
#Tiny Yolo V2 model configuration

using Knet: Knet, progress, progress!, gpu, KnetArray, relu, minibatch, conv4, pool, softmax

import ..sigmoid, ..Settings

mutable struct v2Chain
    layers
    v2Chain(layers...) = new(layers)
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
    x = reshape(x, y.grid_x + 1, y.grid_y + 1, 1, 512 * y.minibatch_size)
    return reshape(conv4(y.w, x; stride = 1), y.grid_x, y.grid_y, 512, y.minibatch_size)
end

(p::Pool)(x) = pool(x; window = p.size, stride = p.stride, padding = p.pad) #pool function

(c::v2Chain)(x) = (for l in c.layers #chain function
    x = l(x)
end; x)

(c::Conv)(x) =
    c.f.(conv4(c.w, x; stride = c.stride, padding = c.padding) .+ c.b) #convolutional layer function

#leaky function
function leaky(x)
    return max(Float32(0.1) * x, x)
end

"""
    load(sets::Settings)

Loads the YOLOv2 model.

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
    # comments are line numbers corresponding
    # with https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg
return v2Chain(
    Conv(3, 3, 3, 32, 1, 1, leaky), #25
    Pool(2, 2, 0), #33
    Conv(3, 3, 32, 64, 1, 1, leaky), #37
    Pool(2, 2, 0), #45
    Conv(3, 3, 64, 128, 1, 1, leaky), #49
    Conv(1, 1, 128, 64, 1, 1, leaky), #57
    Conv(3, 3, 64, 128, 1, 1, leaky), #65
    Pool(2, 2, 0), #73
    Conv(3, 3, 128, 256, 1, 1, leaky), #77
    Conv(1, 1, 256, 128, 1, 1, leaky), #85
    Conv(3, 3, 128, 256, 1, 1, leaky), #93
    Pool(2, 2, 0), #101
    Conv(3, 3, 256, 512, 1, 1, leaky), #105
    Conv(1, 1, 512, 256, 1, 1, leaky), #113
    Conv(3, 3, 256, 512, 1, 1, leaky), #121
    Conv(1, 1, 512, 256, 1, 1, leaky), #129
    Conv(3, 3, 256, 512, 1, 1, leaky), #137
    Pool(2, 2, 0), #145
    Conv(3, 3, 512, 1024, 1, 1, leaky), #149
    Conv(1, 1, 1024, 512, 1, 1, leaky), #157
    Conv(3, 3, 512, 1024, 1, 1, leaky), #165
    Conv(3, 3, 512, 1024, 1, 1, leaky), #173
    Conv(3, 3, 512, 1024, 1, 1, leaky), #181
    ######
    Conv(3, 3, 512, 1024, 1, 1, leaky), #192
    Conv(3, 3, 512, 1024, 1, 1, leaky), #200
    #TODO: [route] #208 https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/cfg/yolov2.cfg#L208
    Conv(1, 1, 128, 64, 1, 1, leaky), #211
    #TODO: [reorg] #219
    #TODO: [route] #222
    Conv(3, 3, 512, 1024, 1, 1, leaky), #225
    Conv(1, 1, 1024, sets.cell_bboxes * (5 + sets.num_classes), 1, 0, identity), #233
)
end

include(joinpath(@__DIR__, "loadweights.jl"))

end #module
