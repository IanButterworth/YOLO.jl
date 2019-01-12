# YOLO.jl
# utils/utils.jl

YOLOsrc = dirname(@__DIR__)

"""
Loads backend handlers
"""
function LoadBackendHandlers()
    ## Backend Handlers
    if isdefined(Main, :Knet) && isdefined(Main, :Flux)
        error("Knet and Flux cannot be loaded at the same time. Restart kernel and load either Flux or Knet with `using` before `using YOLO` to load backend handlers.")
    elseif isdefined(Main, :Knet)
        include(joinpath(YOLOsrc,"backends/Knet.jl")) #Nested in a `quote` to prevent package missing warnings
        println("YOLO: Knet backend handlers loaded.")
    elseif isdefined(Main, :Flux)
        include(joinpath(YOLOsrc,"backends/Flux.jl")) #Nested in a `quote` to prevent package missing warnings
        println("YOLO: Flux backend handlers loaded.")
    else
        @warn "YOLO: No backend loaded. Restart kernel and load either Flux or Knet with `using` before `using YOLO` to load backend handlers."
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
    Returns a triangular matrix of Intersection of Union (IoU) values for the unique & non-self combinations of an array 
    of bounding boxes
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

# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm2d") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)


###### Used once in test.py
# def compute_ap(recall, precision):
#     """ Compute the average precision, given the recall and precision curves.
#     Code originally from https://github.com/rbgirshick/py-faster-rcnn.
#
#     # Arguments
#         recall:    The recall curve (list).
#         precision: The precision curve (list).
#     # Returns
#         The average precision as computed in py-faster-rcnn.
#     """
#     # correct AP calculation
#     # first append sentinel values at the end
#     mrec = np.concatenate(([0.0], recall, [1.0]))
#     mpre = np.concatenate(([0.0], precision, [0.0]))
#
#     # compute the precision envelope
#     for i in range(mpre.size - 1, 0, -1):
#         mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
#
#     # to calculate area under PR curve, look for points
#     # where X axis (recall) changes value
#     i = np.where(mrec[1:] != mrec[:-1])[0]
#
#     # and sum (\Delta recall) * prec
#     ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#     return ap
