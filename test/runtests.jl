using YOLO, Test, BenchmarkTools

enable_info() = Base.CoreLogging.disable_logging(Base.CoreLogging.LogLevel(-1)) #Enable printing of @info messages
disable_info() = Base.CoreLogging.disable_logging(Base.CoreLogging.Info) #disable printing of @info messages

disable_info()
datadir = joinpath(dirname(dirname(pathof(YOLO))), "data")
datasetsdir = joinpath(datadir, "datasets")
pretraineddir = joinpath(datadir, "pretrained")

if !isdir(joinpath(datasetsdir, "voc", "VOCdevkit", "VOC2007"))
    @info "Downloading VOC dataset for testing..."
    YOLO.download_dataset("voc")
end

@testset "Example datasets have been downloaded" begin
    @test isdir(joinpath(datasetsdir, "voc", "VOCdevkit", "VOC2007"))
end

@testset "Example pretrained weights files have been downloaded" begin
    @test isfile(joinpath(
        pretraineddir,
        "v2_tiny",
        "voc",
        "v2_tiny_voc.weights",
    ))
end


@testset "Loading and running YOLOv2_tiny_voc pretrained model" begin

    num_images = 10

    voc = YOLO.datasets.VOC.populate()
    @test length(voc.image_paths) == 5011
    @test length(voc.label_paths) == 5011
    @test length(voc.objects) == 20
    @test voc.objectcounts == [
        306,
        353,
        486,
        290,
        505,
        229,
        1250,
        376,
        798,
        259,
        215,
        510,
        362,
        339,
        4690,
        514,
        257,
        248,
        297,
        324,
    ]

    settings = YOLO.pretrained.v2_tiny_voc.load(minibatch_size = num_images)
    @test settings.num_classes == 20

    vocloaded = YOLO.load(voc, settings, indexes = collect(1:num_images))
    @test size(vocloaded.imagestack_matrix) == (416, 416, 3, num_images)
    @test length(vocloaded.paddings) == num_images
    @test length(vocloaded.labels) == num_images
    #Checks that the VOC download hasn't changed
    @test vec(sum(vocloaded.imagestack_matrix, dims = (1, 2, 3))) ≈ [
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
    ]

    model = YOLO.v2_tiny.load(settings)
    YOLO.loadWeights!(model, settings)

    gt = Float32[
        -0.11728526,
        -0.1913954,
        -0.22067541,
        -0.22349684,
        0.37789905,
        -0.09833967,
        -0.05935463,
        -0.080488496,
        0.17405395,
        -0.11423076,
    ]
    res = model(vocloaded.imagestack_matrix) #run once to do compillation overhead
    t = @elapsed for i = 1:3
        res = model(vocloaded.imagestack_matrix)
        @test res[1, 1, 1, 1:10] == gt
    end

    inference_time = (t / (num_images * 3))
    inference_rate = 1 / inference_time
    @test inference_time < 0.250 #seconds

    enable_info()
    @info "YOLO_v2_tiny inference time per image: $(round(inference_time, digits=2)) seconds ($(round(inference_rate, digits=2)) fps)"
end
