using YOLO
using Test, DataFrames
using FileIO, ImageCore, ImageTransformations

prepareimage(img, w, h) = gpu(reshape(permutedims(Float32.(channelview(imresize(img, w, h))[1:3,:,:]), [3,2,1]), h, w, 3, 1))

pkgdir = dirname(@__DIR__)

modelsAndWeights = [
                ("yolov2-tiny", "yolov2-tiny-COCO", 18),
                # ("yolov2-608", "yolov2-COCO", 0),
                ("yolov3-tiny", "yolov3-tiny-COCO", 2),
                ("yolov3-320", "yolov3-COCO", 1),
                #("yolov3-416", "yolov3-COCO", 3),
                #("yolov3-608", "yolov3-COCO", 1),
                #("yolov3-spp", "yolov3-spp-COCO", 0)
                ]
IMG = load(joinpath(@__DIR__,"images","dog-cycle-car.png"))

df = DataFrame(model=String[], load=Bool[], load_time=Float64[], run=Bool[], forwardpass_time=Float64[], objects_detected=Int64[])
for (i, modelAndWeights) in pairs(modelsAndWeights)
    model = modelAndWeights[1]
    weights = modelAndWeights[2]
    expectedresults = modelAndWeights[3]

    @testset "Model: $model Weights: $weights" begin
        new_df = DataFrame(model=model, load=false, load_time=0.0, run=false, forwardpass_time=0.0, objects_detected=0)
        cfg_file = joinpath(pkgdir, "models", "$(model).cfg")
        weights_file = YOLO.getArtifact(weights)

        t_load = @elapsed begin
            yolomod = YOLO.Yolo(cfg_file, weights_file, 1, silent=true)
        end
        new_df[1, :load] = true
        new_df[1, :load_time] = t_load

        IMG_for_model = prepareimage(IMG, yolomod.cfg[:width], yolomod.cfg[:height])
        res = yolomod(IMG_for_model)
        t_run = @elapsed yolomod(IMG_for_model);
        new_df[1, :run] = true
        new_df[1, :forwardpass_time] = t_run
        new_df[1, :objects_detected] = size(res,2)
        @test size(res,2) == expectedresults
        append!(df, new_df)
        @info "Loaded in $(round(t_load, digits=2)) seconds. Ran in $(round(t_run, digits=2)) seconds."
    end
    GC.gc()
end
display(df)
