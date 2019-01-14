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

"""
    Returns the Intersection of Union (IoU) of two bounding boxes 
    IoU = Area of Overlap / Area of Union
"""
function bbox_iou(box1::Array{Float64,1}, box2::Array{Float64,1}; xywh=true)
    
    if xywh
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b1_y1, b1_y2 = box1[2] - box1[4] / 2, box1[2] + box1[4] / 2
        b2_x1, b2_x2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        b2_y1, b2_y2 = box2[2] - box2[4] / 2, box2[2] + box2[4] / 2
    else
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[1], box1[2], box1[3], box1[4]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[1], box2[2], box2[3], box2[4]
    end
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = maximum([b1_x1, b2_x1])
    inter_rect_y1 = maximum([b1_y1, b2_y1])
    inter_rect_x2 = minimum([b1_x2, b2_x2])
    inter_rect_y2 = minimum([b1_y2, b2_y2])
    # Intersection area
    inter_area = clamp(inter_rect_x2 - inter_rect_x1 + 1, 0, Inf) * clamp(
        inter_rect_y2 - inter_rect_y1 + 1, 0, Inf)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
end

"""
    Returns a triangular matrix of Intersection of Union (IoU) values for the unique 
    & non-self combinations of a single array of bounding boxes
"""
function bbox_iou(bboxes::Array{Array{Float64,1},1}; xywh=true)
    n = size(bboxes,1)
    ious = zeros(Float64,n,n)
    for i = 1:n
        for j  = i+1:n
            ious[i,j] = bbox_iou(bboxes[i],bboxes[j],xywh=xywh)
        end
    end
    
    return ious
end

"""
    Returns a 2D array of Intersection of Union (IoU) values for 
    combinations of two arrays of bounding boxes
"""
function bbox_iou(bboxes1::Array{Array{Float64,1},1},bboxes2::Array{Array{Float64,1},1}; xywh=true)
    n = size(bboxes1,1)
    m = size(bboxes2,1)
    ious = zeros(Float64,n,m)
    for i = 1:n
        for j  = 1:m
            ious[i,j] = bbox_iou(bboxes1[i],bboxes2[j],xywh=xywh)
        end
    end
    
    return ious
end
