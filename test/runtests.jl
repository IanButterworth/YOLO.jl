using YOLO, Test

datadir = joinpath(dirname(dirname(pathof(YOLO))),"data")

# @info "Downloading VOC dataset for testing"
# !isdir(joinpath(datadir,"datasets","voc")) && YOLO.download_dataset("voc")
#
# @testset "Example datasets have been downloaded" begin
#     datasetsdir = joinpath(datadir,"datasets")
#     @test isdir(joinpath(datasetsdir,"voc","VOCdevkit","VOC2007"))
# end

@testset "Example pretrained weights files have been downloaded" begin
    pretraineddir = joinpath(datadir,"pretrained")
    @test isfile(joinpath(pretraineddir,"v2_tiny","voc","v2_tiny_voc.weights"))
end
