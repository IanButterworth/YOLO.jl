const models_dir = joinpath(@__DIR__,"models")
const data_dir = joinpath(@__DIR__,"..","data")
const datasets_dir = joinpath(data_dir,"datasets")
const pretrained_dir = joinpath(data_dir,"pretrained")

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

"""
    sigmoid(x)

Standard sigmoid.
"""
sigmoid(x) = 1.0 / (1.0 .+ exp(-x))
