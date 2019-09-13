module v2_tiny
#Tiny Yolo V2-tiny model configuration

include(joinpath(@__DIR__,"..","common.jl"))
include(joinpath(@__DIR__,"loadweights.jl"))

mutable struct Chain
    layers
    Chain(layers...) = new(layers)
end

mutable struct Conv; w; b; stride; padding; f; end #Define convolutional layer
mutable struct YoloPad; w; end #Define Yolo padding layer (assymetric padding).
struct Pool; size; stride; pad; end # Define pool layer

YoloPad(w1::Int,w2::Int,cx::Int,cy::Int) = YoloPad(zeros(Float32,w1,w2,cx,cy))#Constructor for Yolopad
Conv(w1::Int,w2::Int,cx::Int,cy::Int,st,pd,f) = Conv(randn(Float32,w1,w2,cx,cy),randn(Float32,1,1,cy,1),st,pd,f)#Constructor for convolutional layer

#Assymetric padding function
function(y::YoloPad)(x)
    x = reshape(x,14,14,1,512*MINIBATCH_SIZE)
    return reshape(conv4(y.w,x; stride = 1),13,13,512,MINIBATCH_SIZE)
end

(p::Pool)(x) = pool(x; window = p.size, stride = p.stride, padding=p.pad) #pool function
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x) #chain function
(c::Conv)(x) = c.f.(conv4(c.w,x; stride = c.stride, padding = c.padding) .+ c.b) #convolutional layer function

#leaky function
function leaky(x)
    if gpu() < 0
        return max(convert(Float32,0.1*x),x)
    end
    return max(0.1*x,x)
end

model = Chain(Conv(3,3,3,16,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,16,32,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,32,64,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,64,128,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,128,256,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,256,512,1,1,leaky),
              Pool(2,1,1),
              YoloPad(2,2,1,1),
              Conv(3,3,512,1024,1,1,leaky),
              Conv(3,3,1024,1024,1,1,leaky),
              Conv(1,1,1024,125,1,0,identity))
end
