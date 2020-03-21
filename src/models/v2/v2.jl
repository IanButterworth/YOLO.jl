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

# Define a custom chain of layers:
struct Chain_layers; layers; Chain_layers(args...)=new(args); end
function (c::Chain_layers)(img)
    res_1_17   = c.layers[1](img)
    res_18_25  = c.layers[2](res_1_17)
    res_26_27  = c.layers[3](res_1_17)
    res_28     = reorg(res_26_27, 2)
    res_29     = concat(res_28, res_18_25)
    res_30_31  = c.layers[4](res_29)
end
(c::Chain_layers)(x,y) = nll(c(x),y)

concat(x,y) = (res = cat(x,y,dims=3))
function reorg(x,stride)
    d1,d2,nr_filt,nr_batch = size(x)
    reshape(x,div(d1,stride),div(d2,stride),nr_filt*(stride^2),nr_batch)
end

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
    # with https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg
c1_17 = v2Chain(
    Conv(3, 3, 3, 32, 1, 1, leaky), #25 l1
    Pool(2, 2, 0), #33 l2
    Conv(3, 3, 32, 64, 1, 1, leaky), #37 l3
    Pool(2, 2, 0), #45 l4
    Conv(3, 3, 64, 128, 1, 1, leaky), #49 l5
    Conv(1, 1, 128, 64, 1, 0, leaky), #57 l6
    Conv(3, 3, 64, 128, 1, 1, leaky), #65 l7
    Pool(2, 2, 0), #73 l8
    Conv(3, 3, 128, 256, 1, 1, leaky), #77 l9
    Conv(1, 1, 256, 128, 1, 0, leaky), #85 l10
    Conv(3, 3, 128, 256, 1, 1, leaky), #93 l11
    Pool(2, 2, 0), #101 l12
    Conv(3, 3, 256, 512, 1, 1, leaky), #105 l13
    Conv(1, 1, 512, 256, 1, 0, leaky), #113 l14
    Conv(3, 3, 256, 512, 1, 1, leaky), #121 l15
    Conv(1, 1, 512, 256, 1, 0, leaky), #129 l16
    Conv(3, 3, 256, 512, 1, 1, leaky) #137 l17
    )
c18_24 = v2Chain(
	Pool(2, 2, 0), #145 l18
	Conv(3, 3, 512, 1024, 1, 1, leaky), #149 l19
	Conv(1, 1, 1024, 512, 1, 0, leaky), #157 l20
	Conv(3, 3, 512, 1024, 1, 1, leaky), #165 l21
	Conv(1, 1, 1024, 512, 1, 0, leaky), #173 l22
	Conv(3, 3, 512, 1024, 1, 1, leaky), #181 l23
	######
	Conv(3, 3, 1024, 1024, 1, 1, leaky), #192 l24
	Conv(3, 3, 1024, 1024, 1, 1, leaky) #200 l25
	)
c26_27 = v2Chain(
	Conv(1, 1, 512, 64, 1, 0, leaky)
	)

c30_31 = v2Chain(
	Conv(3, 3,1280 , 1024, 1, 1, leaky), #225 l39
	Conv(1, 1, 1024, sets.cell_bboxes * (5 + sets.num_classes), 1, 0, identity), #233 l31
	)

return Chain_layers(c1_17, c18_24, c26_27, c30_31)

end

include(joinpath(@__DIR__, "loadweights.jl"))

end #module
# ## initial
# return v2Chain(
#     Conv(3, 3, 3, 32, 1, 1, leaky), #25 l1
#     Pool(2, 2, 0), #33 l2
#     Conv(3, 3, 32, 64, 1, 1, leaky), #37 l3
#     Pool(2, 2, 0), #45 l4
#     Conv(3, 3, 64, 128, 1, 1, leaky), #49 l5
#     Conv(1, 1, 128, 64, 1, 1, leaky), #57 l6
#     Conv(3, 3, 64, 128, 1, 1, leaky), #65 l7
#     Pool(2, 2, 0), #73 l8
#     Conv(3, 3, 128, 256, 1, 1, leaky), #77 l9
#     Conv(1, 1, 256, 128, 1, 1, leaky), #85 l10
#     Conv(3, 3, 128, 256, 1, 1, leaky), #93 l11
#     Pool(2, 2, 0), #101 l12
#     Conv(3, 3, 256, 512, 1, 1, leaky), #105 l13
#     Conv(1, 1, 512, 256, 1, 1, leaky), #113 l14
#     Conv(3, 3, 256, 512, 1, 1, leaky), #121 l15
#     Conv(1, 1, 512, 256, 1, 1, leaky), #129 l16
#     Conv(3, 3, 256, 512, 1, 1, leaky), #137 l17
#     Pool(2, 2, 0), #145 l18
#     Conv(3, 3, 512, 1024, 1, 1, leaky), #149 l19
#     Conv(1, 1, 1024, 512, 1, 1, leaky), #157 l20
#     Conv(3, 3, 512, 1024, 1, 1, leaky), #165 l21
#     Conv(1, 1, 1024, 512, 1, 1, leaky), #173 l22
#     Conv(3, 3, 512, 1024, 1, 1, leaky), #181 l23
#     ######
#     Conv(3, 3, 1024, 1024, 1, 1, leaky), #192 l24
#     Conv(3, 3, 1024, 1024, 1, 1, leaky), #200 l25
#     #TODO: [route] #208 https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/cfg/yolov2.cfg#L208 l26
#     Conv(1, 1, 512, 64, 1, 1, leaky), #211 l27
#     #TODO: [reorg] #219 l28
#     #TODO: [route] #222 l29
#     Conv(3, 3,1280 , 1024, 1, 1, leaky), #225 l39
#     Conv(1, 1, 1024, sets.cell_bboxes * (5 + sets.num_classes), 1, 0, identity), #233 l31
# )
