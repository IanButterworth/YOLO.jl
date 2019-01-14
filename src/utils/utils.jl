# YOLO.jl
# utils/utils.jl

YOLOsrc = dirname(@__DIR__)

"""
Loads backend handlers
"""
function LoadBackendHandlers(backend::String)
    ## Backend Handlers
    if backend == "Knet"
        include(joinpath(YOLOsrc,"backends/Knet.jl")) #Nested in a `quote` to prevent package missing warnings
        println("YOLO: Knet backend handlers loaded.")
    elseif backend == "Flux"
        include(joinpath(YOLOsrc,"backends/Flux.jl")) #Nested in a `quote` to prevent package missing warnings
        println("YOLO: Flux backend handlers loaded.")
    else
        @warn "YOLO: Unrecognized backend. Options are Flux or Knet."
    end
end


"""
Loads class labels at 'path'
"""
function load_classes(path)
    classes = readlines(path)
    return classes[classes .!== ""]
end



