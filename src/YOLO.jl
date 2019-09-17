module YOLO

using Knet: Knet, progress, progress!, gpu, KnetArray, relu, minibatch, conv4, pool, softmax
using Random, Glob, FileIO, DelimitedFiles, OffsetArrays
using Images, ImageDraw, ImageFiltering, ImageTransformations, Colors
using FreeTypeAbstraction
using LightXML
using ImageMagick
import ProgressMeter

include("common.jl")

const face = newface(joinpath(@__DIR__,"misc","DroidSansMono.ttf")) #Font type
const xtype=(Knet.gpu()>=0 ? Knet.KnetArray{Float32} : Array{Float32})#if gpu exists run on gpu

Base.@kwdef mutable struct Label
    x::Float32 #Left hand edge in scaled image width unnits (0-1)
    y::Float32 #Right hand edge in scaled image width unnits (0-1)
    w::Float32 #Width in scaled image width unnits (0-1)
    h::Float32 #Height in scaled image height unnits (0-1)
    name::String #Description of object
    difficult::Int32 #Difficult flag
end

Base.@kwdef mutable struct LabelledImageDataset
    name::String
    objects::Vector{String}
    objectcounts::Vector{Int32}
    image_size_lims::Tuple{Int,Int} = (-1,-1)
    images_dir::String
    labels_dir::String
    image_paths::Array{String} = String[]
    label_paths::Array{String} = String[]
    labels::Vector{Vector{Label}} = Vector{Vector{Label}}(undef,0)
end

Base.@kwdef mutable struct Settings
    dataset_description::String = ""
    source::String = ""
    image_shape::Tuple{Int,Int}
    image_channels::Int
    namesdic::Dict{String,Int64}
    numsdic::Dict{Int64,String}
    anchors::Array{Tuple{Float64,Float64}}
    num_classes::Int
end

Base.@kwdef mutable struct LoadedDataset
    imagestack_matrix::Array{Float32}                       #4D image stack of type Float32 (w,h,colorchannels,numimages)
    paddings::Vector{Array{Int}}
    labels::Vector{Vector{Label}}
    #label #labels as tupple of arrays. tupples are designed as (ImageWidth, ImageHeight,[x,y,objectWidth,objectHeight],[x,y,objectWidth,objectHeight]..)
end

include("datasets.jl")
include("pretrained.jl")
include("models.jl")
include("preprocess.jl")
include("postprocess.jl")


"""
    load_v2_tiny_voc(;output_dir::String = "YOLO_output")

Refactored based on yolov2 by Yavuz Bakman,2019. Tiny Yolo V2 implementation by Knet framework
"""
function load_v2_tiny_voc(;output_dir::String = "YOLO_output")

    minibatch_size = 1
    weights_file = joinpath(pretrained_dir,"v2_tiny","voc","yolov_tiny-_oc.weights") #Pre-trained weights data

    #Load pre-trained weights into the model
    f = open(weights_file)
    loadweights!(model,f)
    close(f)


    EXAMPLE_INPUT = joinpath(output_dir,"example.jpg") #One input for display

    totaldic = createcountdict(v2_tiny_voc.namesdic)

end

function batchaccuracy()
    #=User guide
    inputandlabelsdir => takes Voc labels and inputs folder location respectively and returns 2 arrays
    images directories and their labels' directories.

    prepareinputlabels => takes images and labels directories and returns 3 arrays
    input as 416*416*3*totalImage and
    labels as tupple of arrays.
    tupples are designed as (ImageWidth, ImageHeight,[x,y,objectWidth,objectHeight],[x,y,objectWidth,objectHeight]..)
    images are padded version of given images.

    inputdir => takes the directory of input for saving output and returns array of directories of the images
    prepareinput => takes the array of directories of the images and returns 416*416*3*totalImage. =#
    #prepare data for accuracy
    images,labels = collectLabelImagePairs(dataset_labels_dir,dataset_images_dir)
    inp,out,imgs = prepareinputlabels(images,labels)
    print("input for accuracy:  ")
    println(summary(inp))
    #Minibatching process
    accdata = minibatch(inp,out,minibatch_size;xtype = xtype)
    AP = accuracy(model,accdata,0.0,o[:iou],o[:iouth])
    display(AP)
    print("Mean average precision: ")
    println(calculatemean(AP))
    if o[:record] == true
        drawdata = minibatch(inp,imgs,minibatch_size; xtype = xtype)
        #output of Voc dataset.
        #return output as [ [(x,y,width,height,classNumber,confidenceScore),(x,y,width,height,classNumber,confidenceScore)...] ,[(x,y,width,height,classNumber,confidenceScore),(x,y,width,height,classNumber,confidenceScore)..],...]
        #save the output images into given location
        result = saveoutput(model,drawdata,o[:confth],o[:iou]; record = true, location = "VocResult")
    end
end

function displayout()
    #Display one test image
    displaytest(EXAMPLE_INPUT,model; record = o[:record])
end
function saveout()
    #prepare data for saving process
    indir = readdir(output_dir)
    inp,images = prepareinput(indir)
    print("input for saving:  ")
    println(summary(inp))
    #Minibatching process
    savedata = minibatch(inp,images,minibatch_size; xtype = xtype)
    #output of given input.
    #return output as same with above example
    #It also saves the result of the images into output folder.
    @time result2 = saveoutput(model,savedata,0.3,0.3; record = o[:record], location = o[:directory])
end
end # module
