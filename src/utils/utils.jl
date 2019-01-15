# YOLO.jl
# utils/utils.jl

using Colors, PyPlot, Clustering, Statistics

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


"""
calculate_KmeanClusters(wh::Array{Float64,2},k::Int64)

Calculates K-mean method (using median) clusters of a wh array (width & height) to determine best input bbox sizes
for training
"""
function calculate_KmeanClusters(wh::Array{Float64,2},k::Int64)
    R = kmeans(collect(wh'),k)   
    clusters = map(i->(median(wh[assignments(R).==i,1]),median(wh[assignments(R).==i,2])),1:k)
    return clusters, assignments(R)
end

"""
visualize_KmeanClusters(wh::Array{Float64,2},clusters::Array{Tuple{Float64,Float64},1},cluster_assignments::Array{Int,1})

Graph the output of  calculate_KmeanClusters(wh::Array{Float64,2},k::Int64)
"""
function visualize_KmeanClusters(wh::Array{Float64,2},clusters::Array{Tuple{Float64,Float64},1},cluster_assignments::Array{Int,1})
    cols = distinguishable_colors(length(clusters)+1, [RGB(1,1,1)])[2:end]
    pcols = map(col -> (red(col), green(col), blue(col)), cols)
    
    
    
    scatter(wh[:,1],wh[:,2],s=0.1,c=pcols[cluster_assignments])
    xlabel("Width");ylabel("Height");title("BBOX size")

    for i = 1:length(clusters)
        scatter(clusters[i][1],clusters[i][2],s=50,color=pcols[i],label=round.(clusters[i],digits=3))
    end
    legend()
end

