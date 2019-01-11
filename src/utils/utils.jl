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
