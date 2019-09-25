using YOLO, Test

enable_info() = Base.CoreLogging.disable_logging(Base.CoreLogging.LogLevel(-1)) #Enable printing of @info messages
disable_info() = Base.CoreLogging.disable_logging(Base.CoreLogging.Info) #disable printing of @info messages

disable_info()
datadir = joinpath(dirname(dirname(pathof(YOLO))), "data")
datasetsdir = joinpath(datadir, "datasets")
pretraineddir = joinpath(datadir, "pretrained")

if !isdir(joinpath(datasetsdir, "voc2007", "VOCdevkit", "VOC2007"))
    @info "Downloading VOC dataset for testing..."
    YOLO.download_dataset("voc2007")
end

@testset "Example datasets have been downloaded" begin
    @test isdir(joinpath(datasetsdir, "voc2007", "VOCdevkit", "VOC2007"))
end

@testset "Example pretrained weights files have been downloaded" begin
    @test isfile(joinpath(
        pretraineddir,
        "v2_tiny",
        "voc2007",
        "v2_tiny_voc.weights",
    ))
end


@testset "Loading and running YOLOv2_tiny_voc pretrained model" begin

    num_images = 2

    voc = YOLO.datasets.VOC.populate()
    @test length(voc.image_paths) == 5011
    @test length(voc.label_paths) == 5011
    @test length(voc.objects) == 20

    settings = YOLO.pretrained.v2_tiny_voc.load(minibatch_size = num_images)
    @test settings.num_classes == 20

    vocloaded = YOLO.load(voc, settings, indexes = collect(1:num_images))
    @test size(vocloaded.imstack_mat) == (416, 416, 3, num_images)
    @test length(vocloaded.paddings) == num_images
    @test length(vocloaded.labels) == num_images
    #Checks that the VOC download hasn't changed
    @test vec(sum(vocloaded.imstack_mat, dims = (1, 2, 3))) ≈ [
        140752.47,
        122024.16,
        126477.125,
        114651.555,
        143701.72,
        196821.47,
        179382.5,
        83476.43,
        132382.72,
        166430.16,
    ][1:num_images]

    model = YOLO.v2_tiny.load(settings)
    YOLO.loadWeights!(model, settings)

    res = model(vocloaded.imstack_mat) #run once to deal with compillation overhead
    #accdata = minibatch(inp, out, settings.minibatch_size; xtype = xtype)

    t = @elapsed for i in 1:10
        model(vocloaded.imstack_mat)
    end

    inference_time = (t / 10) / num_images
    inference_rate = 1 / inference_time

    predictions = YOLO.postprocess(res, settings, conf_thresh = 0.3, iou_thresh = 0.3)
    @test length(predictions) == num_images # images
    @test length(predictions[1]) == 0
    @test length(predictions[2]) == 1

    @test isapprox(predictions[2][1].bbox.x, 0.23406872f0, rtol=0.05)
    @test isapprox(predictions[2][1].bbox.y, 0.28794688f0, rtol=0.05)
    @test isapprox(predictions[2][1].bbox.w, 0.8048826f0, rtol=0.05)
    @test isapprox(predictions[2][1].bbox.h, 0.46979588f0, rtol=0.05)
    @test predictions[2][1].class == 7
    @test isapprox(predictions[2][1].conf, 0.5042128f0, rtol=0.05)

    t = @elapsed for i in 1:10
        YOLO.postprocess(res, settings, conf_thresh = 0.3, iou_thresh = 0.3)
    end

    postprocess_time = (t / 10) / num_images
    postprocess_rate = 1 / postprocess_time

    ## Makie Tests
    #disabled because Makie can't be tested on headless CI
    # scene = YOLO.renderResult(vocloaded.imstack_mat[:,:,:,1], predictions[1], settings, save_file = "test.png")
    # @test isfile("test.png")
    # rm("test.png", force=true)

    enable_info()
    @info "YOLO_v2_tiny inference time per image: $(round(inference_time, digits=4)) seconds ($(round(inference_rate, digits=2)) fps)"
    @info "YOLO_v2_tiny postprocess time per image: $(round(postprocess_time, digits=4)) seconds ($(round(postprocess_rate, digits=2)) fps)"
    @info "Total time per image: $(round(inference_time + postprocess_time, digits=4)) seconds ($(round(1/(inference_time + postprocess_time), digits=2)) fps)"

end
