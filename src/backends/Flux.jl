# YOLO.jl
# Flux.jl

# Flux backend handlers to convert the generalized layers to Flux

# By design, this script is only loaded when Flux is loaded in Main
## It would be great to not have Flux as a dependency, given option of Flux OR Knet
## NEEDS TO BE FIXED (like `Plots.jl` has multiple backend options)

using Flux

Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity; init = Flux.glorot_uniform,  stride = 1, pad = 0, dilation = 1) where N = Flux.Conv(k, ch, σ, init = init, stride = stride, pad = pad, dilation = dilation)

BatchNorm(args...) = Flux.BatchNorm(args...)

MaxPool(x::AbstractArray, k; pad = map(_->0,k), stride = k) = Flux.maxpool(x,k,pad=pad,stride=stride)
    
export YOLOConv, YOLOBatchNorm, YOLOMaxPool


