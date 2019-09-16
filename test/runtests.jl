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

@testset "Loading VOC dataset" begin
    voc = YOLO.VOC.populate()
    @test length(voc.image_paths) == 5011
    @test length(voc.label_paths) == 5011
    @test length(voc.objects) == 20
    @test voc.objectcounts == [306, 353, 486, 290, 505, 229, 1250, 376, 798, 259, 215, 510, 362, 339, 4690, 514, 257, 248, 297, 324]

    sets = YOLO.Settings(image_shape=(416,416),image_channels=3)

    vocloaded = YOLO.load(voc, sets, limitfirst=10)
    @test size(vocloaded.imagestack_matrix) ==  (416,416,3,10)
    @test length(vocloaded.paddings) == 10
end
