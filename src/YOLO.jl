module YOLO

using Knet: Knet, progress, progress!, gpu, KnetArray, relu, minibatch, conv4, pool, softmax
using Random, Glob, FileIO, DelimitedFiles, OffsetArrays
using Images, ImageDraw, ImageFiltering, ImageTransformations, Colors
using FreeTypeAbstraction
using LightXML
using ImageMagick

const models_dir = joinpath(@__DIR__,"models")
const data_dir = joinpath(@__DIR__,"..","data")
const datasets_dir = joinpath(data_dir,"datasets")
const pretrained_dir = joinpath(data_dir,"pretrained")

const face = newface(joinpath(@__DIR__,"misc","DroidSansMono.ttf")) #Font type
const xtype=(Knet.gpu()>=0 ? Knet.KnetArray{Float32} : Array{Float32})#if gpu exists run on gpu

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
    getweights(model,f)
    close(f)

    ## Dataset for validation testing
    dataset_images_dir = joinpath(datasets_dir,"voc","VOCdevkit","VOC2007","JPEGImages") #Input directory for accuracy calculation
    dataset_labels_dir = joinpath(datasets_dir,"voc","VOCdevkit","VOC2007","Annotations") #location of objects as Xml file for accuracy calculation


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
    images,labels = inputandlabelsdir(dataset_labels_dir,dataset_images_dir)
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
    indir = inputdir(output_dir)
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
