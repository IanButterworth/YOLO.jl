using YOLO, Test

Base.CoreLogging.disable_logging(Base.CoreLogging.Info) #disable printing of @info messages

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

@testset "Populating VOC dataset" begin
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
end

@testset "Loading pretrained models" begin
    settings = YOLO.pretrained.v2_tiny_voc.load()

    @test settings.num_classes == 20

    #model = YOLO.v2.load(settings)
    #loadWeights!(model, weightsfile)
end

@testset "Loading VOC model based on pretrained settings" begin
    voc = YOLO.datasets.VOC.populate()
    settings = YOLO.pretrained.v2_tiny_voc.load()
    vocloaded = YOLO.load(voc, settings, indexes = collect(1:10))
    @test size(vocloaded.imagestack_matrix) == (416, 416, 3, 10)
    @test length(vocloaded.paddings) == 10
    @test length(vocloaded.labels) == 10

    #This test checks that the VOC download hasn't changed
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
end
