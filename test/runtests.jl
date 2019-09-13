using YOLO, Test

datadir = joinpath(dirname(dirname(pathof(YOLO))),"data")

@testset "Example pretrained weights files have been downloaded" begin

    pretraineddir = joinpath(datadir,"pretrained")
    @test isfile(joinpath(pretraineddir,"v2_tiny","voc","v2_tiny_voc.weights"))
end

@testset "Example datasets have been downloaded" begin
    datasetsdir = joinpath(datadir,"datasets")
    @test isdir(joinpath(datasetsdir,"voc","VOCdevkit","VOC2007"))
end
