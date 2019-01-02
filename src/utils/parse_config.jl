# FluxYOLOv3.jl
# utils/parse_config.jl

"""
Parses the yolo-v3 layer configuration file and returns module definitions
"""
function parse_model_config(path::String)
    lines = readlines(path)
    lines = strip.(lines[.!startswith.(lines,"#")]) # remove comments and get rid of fringe whitespaces

    module_defs = Array{Dict}(undef,0)
    d = Dict()
    for line in lines
        if startswith(line,"[") # This marks the start of a new block
            d = Dict("type"=>string(strip(line,['[',']'])))
            if d["type"] == "convolutional"
                push!(d,("batch_normalize"=>"0"))
            end
        else
            elems = split(line,"=")
            if length(elems) == 2
                key = string(strip(elems[1]))
                value = string(strip(elems[2]))
                push!(d,(key=>value))
            end
        end
        push!(module_defs,d)
    end

    return module_defs
end

"""
Parses the data configuration file
"""
function parse_data_config(path::String)
    options = Dict()
    push!(options,("gpus"=>"0,1,2,3"))
    push!(options,("num_workers"=>"10"))
    lines = readlines(path)
    lines = strip.(lines[.!startswith.(lines,"#")]) # remove comments and get rid of fringe whitespaces
    lines = lines[lines !== ""] # remove empty lines
    for line in lines
        elems = split(line,"=")
        if length(elems) == 2
            key = string(strip(elems[1]))
            value = string(strip(elems[2]))
            push!(options,(key=>value))
        end
    end
    return options
end
