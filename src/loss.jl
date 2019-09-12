## Based on the yolov3 paper and https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
using Flux: onehot

struct YOLOConfig
    # For an image
    cells::Int = 7 # Number of cells in x and y (SxS grid). Should be odd to give cell in center
    # For each grid cell
    bboxes::Int = 2 # Number of BBOXes
    objects::Int = 1 # Number of objects (fixed)
    classes::Int = 20 # Number of classses
end

struct BBox{T} where T<:AbstractFloat
    x::T #Offset of center of bbox in x (scaled to image width (0-1)) from the top left corner of the cell
    y::T # = Offset of center of bbox in y (scaled to image width (0-1)) from the top left corner of the cell
    w::T # = Width (scaled to image width (0-1))
    h::T # = Height (scaled to image height (0-1))
end
struct Truth{N,M} where N<:AbstractFloat, M<:Integer
    bbox::BBox{N}
    class::M
end
"""
    loss(pred::Array{Float32},truth::Array{Float32})

    size(pred) == (S, S, B×5 + C)
    Each boundary box contains 5 elements: (x, y, w, h) and a box confidence score.
    The confidence score reflects how likely the box contains an object (objectness)
    and how accurate is the boundary box. We normalize the bounding box width w
    and height h by the image width and height. x and y are offsets to the
    corresponding cell. Hence, x, y, w and h are all between 0 and 1. Each cell
    has 20 conditional class probabilities. The conditional class probability is
    the probability that the detected object belongs to a particular class (one
    probability per category for each cell). So, YOLO’s prediction has a shape
    of (S, S, B×5 + C) = (7, 7, 2×5 + 20) = (7, 7, 30).

    Bounding Boxes (bbox)
    x = Offset of center of bbox in x (scaled to image width (0-1)) from the top left corner of the cell
    y = Offset of center of bbox in y (scaled to image width (0-1)) from the top left corner of the cell
    w = Width (scaled to image width (0-1))
    h = Height (scaled to image height (0-1))

"""
function loss(preds::Array{Float32,3}, truths::Array{Vector{Truth},2}, config::YOLOConfig;
    λcoord::Float32=10.0,
    λnoobj::Float32=0.5
    obj_threshold::Float32=0.5.
    )

    for xi in 1:S, yi in 1:S    #Iterate through cells
        pred_classes = view(preds,config.cells,config.cells,((config.bboxes*5)+1):((config.bboxes*5)+1)+config.classes)
        for truth in truths[xi,yi]  #Iterate through truth bboxes (could be multiple per cell)
            onehot(truth.class,1:config.classes)
            truth.bbox


        # Classification loss
        loss_classification[xi,yi] = sum((truth_classes - pred_classes).^2)

        for bi in 1:B
            pred_bbox = view(pred,S,S,(1:4).+((bi-1)*5))
            pred_objectness = view(pred,S,S,(bi-1)*5)

            obj = (pred_objectness >= obj_threshold) ? 1 : 0



            # Localization loss (errors between the predicted boundary box and the ground truth)
            loss_localization =

            # Confidence loss (objectness loss)
            loss_confidence =
        end
    end

    # Localization loss
    l = (λcoord * sum(loss_localization))
    l += (λcoord * sum(loss_shape))

    # Confidence loss
    l += (λnoobj * sum(loss_confidence))

    # Classification loss
    l += (λcoord * sum(loss_classification))
    return l
end
