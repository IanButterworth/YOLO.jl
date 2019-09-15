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
