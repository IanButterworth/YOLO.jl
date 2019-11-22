const models_dir = joinpath(dirname(@__DIR__), "models")

getArtifact(name::String) = joinpath(@artifact_str(name), "$(name).weights")

"""
    flipdict(dict::Dict)

Flip the key=>value pair for each entry in a dict.
"""
flipdict(dict::Dict) = Dict(map(x->(dict[x],x),collect(keys(dict))))

"""
    createcountdict(dict::Dict)

Create a dict copy of namesdict, for counting the occurances of each named object.
"""
createcountdict(dict::Dict) = Dict(map(x->(x,0),collect(keys(dict))))

emptybatch(mod::Yolo) = Flux.gpu(Array{Float32}(undef, mod.cfg[:width], mod.cfg[:height], mod.cfg[:channels], mod.cfg[:batchsize]))

function benchmark(;select = [1,3,4])
    pretrained_list = [
                        YOLO.v2_tiny_416_COCO,
                        YOLO.v2_608_COCO,
                        YOLO.v3_tiny_416_COCO,
                        YOLO.v3_320_COCO,
                        YOLO.v3_416_COCO,
                        YOLO.v3_608_COCO,
                        YOLO.v3_608_spp_COCO
                        ][select]

    IMG = rand(RGB,416,416)

    header = ["Model" "loaded?" "load time (s)" "ran?" "run time (s)" "objects detected"]
    table = Array{Any}(undef, length(pretrained_list), 6)
    for (i, pretrained) in pairs(pretrained_list)
        modelname = string(pretrained)
        @info "Loading and running $modelname"
        table[i,:] = [modelname false "-" "-" "-" "-"]

        t_load = @elapsed begin
            yolomod = pretrained(silent=true)
        end
        table[i, 2] = true
        table[i, 3] = round(t_load, digits=3)

        batch = YOLO.emptybatch(yolomod)
        batch[:,:,:,1] .= YOLO.gpu(resizePadImage(IMG, yolomod))

        res = yolomod(batch) #run once
        t_run = @belapsed $yolomod($batch);
        table[i, 4] = true
        table[i, 5] = round(t_run, digits=4)
        table[i, 6] = size(res,2)

        GC.gc()
    end
    pretty_table(table, header)
end
