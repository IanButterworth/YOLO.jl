using YOLO
using Test, PrettyTables
using FileIO, ImageCore, ImageTransformations

prepareimage(img, w, h) = YOLO.Flux.gpu(reshape(permutedims(Float32.(channelview(imresize(img, w, h))[1:3,:,:]), [3,2,1]), h, w, 3, 1))

pkgdir = dirname(@__DIR__)

pretrained_list = [
                    YOLO.v2_tiny_416_COCO,
                    # YOLO.v2_608_COCO,
                    YOLO.v3_tiny_416_COCO,
                    YOLO.v3_320_COCO,
                    # YOLO.v3_416_COCO,
                    # YOLO.v3_608_COCO,
                    #YOLO.v3_608_spp_COCO
                    ]

IMG = load(joinpath(@__DIR__,"images","dog-cycle-car.png"))

header = ["Model" "loaded?" "load time (s)" "ran?" "run time (s)" "objects detected"]
table = Array{Any}(undef, length(pretrained_list), 6)
for (i, pretrained) in pairs(pretrained_list)
    modelname = string(pretrained)
    @testset "Pretrained Model: $modelname" begin
        global table
        table[i,:] = [modelname false "-" "-" "-" "-"]

        t_load = @elapsed begin
            yolomod = pretrained(silent=true)
        end
        table[i, 2] = true
        table[i, 3] = round(t_load, digits=3)
        @info "Loaded in $(round(t_load, digits=2)) seconds."

        #IMG_for_model = prepareimage(IMG, yolomod.cfg[:width], yolomod.cfg[:height])
        batch = YOLO.emptybatch(yolomod)
        batch[:,:,:,1] .= YOLO.gpu(resizePadImage(IMG, yolomod))

        yolomod(batch) #run once
        val, t_run, bytes, gctime, memallocs = @timed res = yolomod(batch);
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
