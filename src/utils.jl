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
