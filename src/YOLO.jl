module YOLO
export Yolo, resizePadImage, resizekern, sizethatfits
# Largely based on Robert Luciani's IMPLEMENTATION OF YOLO: https://github.com/r3tex/ObjectDetector.jl

using Flux
import Flux.gpu
using CuArrays
using CUDAnative

CuArrays.allowscalar(false)
CuFunctional = CUDAnative.functional()

using Pkg.Artifacts

using ImageFiltering
using ImageTransformations
using ImageCore

using BenchmarkTools
using PrettyTables

include("core.jl")
include("pretrained.jl")

include("prepareimage.jl")
include("utils.jl")

end #module
