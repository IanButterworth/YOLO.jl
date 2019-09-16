using YOLO, Test

datadir = joinpath(dirname(dirname(pathof(YOLO))),"data")
datasetsdir = joinpath(datadir,"datasets")
pretraineddir = joinpath(datadir,"pretrained")

if !isdir(joinpath(datasetsdir,"voc","VOCdevkit","VOC2007"))
    @info "Downloading VOC dataset for testing..."
    YOLO.download_dataset("voc")
end

@testset "Example datasets have been downloaded" begin
    @test isdir(joinpath(datasetsdir,"voc","VOCdevkit","VOC2007"))
end

@testset "Example pretrained weights files have been downloaded" begin
    @test isfile(joinpath(pretraineddir,"v2_tiny","voc","v2_tiny_voc.weights"))
end

@testset "Loading VOC dataset" being
    voc = YOLO.VOC.populate()
    @test length(voc.image_paths) == 5011
    @test length(voc.label_paths) == 5011
    @test voc.objectcounts == 20

    sets = YOLO.Settings(image_shape=(416,416),image_channels=3)

    vocloaded = YOLO.load(voc, sets)
    @test size(vocloaded.imagestack_matrix) ==  [416,416,3,5011]
    @test length(paddings) = 5011
end
