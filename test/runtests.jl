using YOLO
using Test, PrettyTables
using FileIO, ImageCore, ImageTransformations

prepareimage(img, w, h) = gpu(reshape(permutedims(Float32.(channelview(imresize(img, w, h))[1:3,:,:]), [3,2,1]), h, w, 3, 1))

pkgdir = dirname(@__DIR__)

modelsAndWeights = [
                ("yolov2-tiny", "yolov2-tiny-COCO", 18),
                # ("yolov2-608", "yolov2-COCO", 0),
                ("yolov3-tiny", "yolov3-tiny-COCO", 3),
                ("yolov3-320", "yolov3-COCO", 2),
                #("yolov3-416", "yolov3-COCO", 3),
                #("yolov3-608", "yolov3-COCO", 1),
                #("yolov3-spp", "yolov3-spp-COCO", 0)
                ]
IMG = load(joinpath(@__DIR__,"images","dog-cycle-car.png"))

header = ["Model" "loaded?" "load time (s)" "ran?" "run time (s)" "objects detected"]
table = Array{Any}(undef,length(modelsAndWeights),6)
for (i, modelAndWeights) in pairs(modelsAndWeights)
    model = modelAndWeights[1]
    weights = modelAndWeights[2]
    expectedresults = modelAndWeights[3]

    @testset "Model: $model Weights: $weights" begin
        global table
        table[i,:] = [model false "-" "-" "-" "-"]
        cfg_file = joinpath(pkgdir, "models", "$(model).cfg")
        weights_file = YOLO.getArtifact(weights)

        t_load = @elapsed begin
            yolomod = YOLO.Yolo(cfg_file, weights_file, 1, silent=true)
        end
        table[i, 2] = true
        table[i, 3] = round(t_load, digits=3)
        @info "Loaded in $(round(t_load, digits=2)) seconds."

        IMG_for_model = prepareimage(IMG, yolomod.cfg[:width], yolomod.cfg[:height])
        yolomod(IMG_for_model)
        val, t_run, bytes, gctime, memallocs = @timed res = yolomod(IMG_for_model);
        table[i, 4] = true
        table[i, 5] = round(t_run, digits=4)
        table[i, 6] = size(res,2)
        #@test size(res,2) == expectedresults
        @test size(res,2) > 0

        @info "Ran in $(round(t_run, digits=2)) seconds. (bytes $bytes, gctime $gctime)"
    end
    GC.gc()
end
pretty_table(table, header)
