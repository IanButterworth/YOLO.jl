# YOLO.jl
# loss.jl
# SOME DEFAULT SETTINGS
# ANCHORS = [1.07709888,  1.78171903, 2.71054693,  5.12469308,  10.47181473,
#                 10.09646365, 5.48531347,  8.11011331]
# LABELS = ["aeroplane","bicycle","bird","boat","bottle",
#                 "bus","car","cat","chair","cow",
#                 "diningtable","dog","horse","motorbike","person",
#                 "pottedplant","sheep","sofa","train","tvmonitor"]
# BATCH_SIZE        = 1 # default was 500
# IMAGE_H, IMAGE_W  = 416, 416
# GRID_H,  GRID_W   = 13 , 13
# TRUE_BOX_BUFFER   = 50
# BOX               = Int(length(ANCHORS)/2)
# CLASS             = length(LABELS)
# LAMBDA_NO_OBJECT = 1.0
# LAMBDA_OBJECT    = 5.0
# LAMBDA_COORD     = 1.0
# LAMBDA_CLASS     = 1.0
# GRIDSIZE         = (GRID_H,  GRID_W) #usually
#
# # VOC folder paths
# train_image_folder = "C:/Framework/Julia/YOLO_loss/VOCdevkit/VOC2012/JPEGImages/"
# #train_image_folder = "C:/Framework/Julia/YOLO_loss/VOCdevkit/VOC2007/JPEGImages/"
# train_annot_folder = "C:/Framework/Julia/YOLO_loss/VOCdevkit/VOC2012/Annotations/"
# #train_annot_folder = "C:/Framework/Julia/YOLO_loss/VOCdevkit/VOC2007/Annotations/"
#
# # Just a nice dict
# generator_config = Dict(
#     "IMAGE_H"         => IMAGE_H,
#     "IMAGE_W"         => IMAGE_W,
#     "GRID_H"          => GRID_H,
#     "GRID_W"          => GRID_W,
#     "LABELS"          => LABELS,
#     "ANCHORS"         => ANCHORS,
#     "BATCH_SIZE"      => BATCH_SIZE,
#     "TRUE_BOX_BUFFER" => TRUE_BOX_BUFFER,
# )
#
# # Stuff used for basic testing
# b_batch = zeros(Float32,1,1,1,1,50,4)# this is just a placeholder for true_boxes dimensions
# true_boxes = zeros(size(b_batch))
# y_batch = zeros(Float32,1, 13, 13, 4, 25)
# sz   = BATCH_SIZE*GRID_W*GRID_H*BOX*(4 + 1 + CLASS)
# #y_pred = np.random.normal(size=size,scale = 10/(GRID_H*GRID_W))
# y_pred = randn(Float32,sz)/33
# y_pred = reshape(y_pred,BATCH_SIZE,GRID_H,GRID_W,BOX,4 + 1 + CLASS)
# y_true = y_pred
# custom_loss_core(y_pred,y_pred,true_boxes,GRIDSIZE,
#         BATCH_SIZE,ANCHORS,LAMBDA_COORD,LAMBDA_CLASS,LAMBDA_NO_OBJECT, LAMBDA_OBJECT)
#
# Basic helper functions
"""
```
sigmoid(x) = 1 / (1 + exp(-x))
```
"""
sigmoid(x) = 1 / (1 + exp(-x))

"""
```
sparse_softmax_cross_entropy_with_logits based on:
https://stackoverflow.com/questions/37775596/where-is-the-origin-coding-of-sparse-softmax-cross-entropy-with-logits-function
```
"""
function sparse_softmax_cross_entropy_with_logits(true_box_class, pred_box_class)
    x = -1.0 .* log.(softmax(pred_box_class)) # tensorflow gives different results
    # for softmax applied on pred_box_class(why???)
    return x[true_box_class]
end

"""
```
expand_dims(x::Array,axis::Int)
Add a new 1-sized dim at axis value, using python style -ve indexing for positions from end of dims
i.e.
size(expand_dims(rand(10,24,12),2))
(10, 1, 24, 12)
size(expand_dims(rand(10,24,12),2))
(10, 24, 1, 12)
size(expand_dims(rand(10,24,12),-1))
(10, 24, 12, 1)
size(expand_dims(rand(10,24,12),-2))
(10, 24, 1, 12)
```
"""
function expand_dims(x::Array,axis::Int64)
    dims = collect(size(x))
    if abs(axis) > length(dims)
        error(string("`axis` value (",abs(axis),") is larger than dimensions of `x` (",length(dims),")"))
    end
    if axis > 0
        splice!(dims, axis, [1,dims[axis]])
        elseif axis < 0
        splice!(dims, length(dims)+axis+1, [dims[length(dims)+axis+1],1])
    else
        error("Axis cannot be equal to 0")
    end

    reshape(x, dims...)
end


# Implementation based on https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html
"""
```
get_cell_grid(gridsize::Tuple{Int,Int},nbatchsize::Int,nboxes::Int)
Helper function to assure that the bounding box x and y are in the grid cell scale
```
"""
function get_cell_grid(gridsize::Tuple{Int,Int},batchsize::Int,nboxes::Int)
    # gridsize = (13,13) #usually
    # batchsize = 1 # adjust based on RAM size
    # nboxes = 4 #(in this case) length(ANCHORS)/2 in python
    cell_grid = Array{Float32}(undef,batchsize,gridsize[1],gridsize[2],nboxes,2)
    for batch in 1:batchsize
        for gh in 1:gridsize[1]
            for gw in 1:gridsize[2]
                for box in 1:nboxes
                    cell_grid[batch,gh,gw,box,1] =  gw - 1
                    cell_grid[batch,gh,gw,box,2] =  gh - 1
                end
            end
        end
    end
    return cell_grid
end

"""
```
Adjust prediction
== input ==
y_pred : takes any real values
tensor of shape = (N batch, NGrid h, NGrid w, NAnchor, 4 + 1 + N class)
ANCHORS : list containing width and height specializaiton of anchor box
== output ==
pred_box_xy : shape = (N batch, N grid x, N grid y, N anchor, 2), contianing [center_y, center_x] rangining [0,0]x[grid_H-1,grid_W-1]
pred_box_xy[irow,igrid_h,igrid_w,ianchor,0] =  center_x
pred_box_xy[irow,igrid_h,igrid_w,ianchor,1] =  center_1
calculation process:
sigmoid(y_pred[...,:2]) : takes values between 0 and 1
sigmoid(y_pred[...,:2]) + cell_grid : takes values between 0 and grid_W - 1 for x coordinate
takes values between 0 and grid_H - 1 for y coordinate
pred_Box_wh : shape = (N batch, N grid h, N grid w, N anchor, 2),
containing width and height, rangining [0,0]x[grid_H-1,grid_W-1]
pred_box_conf : shape = (N batch, N grid h, N grid w, N anchor, 1), containing confidence to range between 0 and 1
pred_box_class : shape = (N batch, N grid h, N grid w, N anchor, N class), containing
```
"""
function adjust_scale_prediction(y_pred::Array{Float32,5}, cell_grid::Array{Float32,5}, anchors::Vector{Float64})
    # anchors is an array with width,height,width,height....
    # size(y_pred) = (1,13,13,4,25)
    nboxes = Int(length(anchors)/2)

    ### adjust x and y
    # the bounding box bx and by are rescaled to range between 0 and 1 for given gird.
    # Since there are nboxes x nboxes grids, we rescale each bx and by to range between 0 to nboxes + 1
    pred_box_xy = sigmoid.(y_pred[:,:,:,:,1:2]) + cell_grid # bx, by
    #size(pred_box_xy) = (1,13,13,4,2)

    ### adjust w and h
    # exp to make width and height positive
    # rescale each grid to make some anchor "good" at representing certain shape of bounding box
    anchor_aux = permutedims(reshape(anchors,1,1,1,2,nboxes),[1,2,3,5,4]) # Julia uses column major
    pred_box_wh = exp.(y_pred[:,:,:,:,3:4]) .* anchor_aux
    #size(pred_box_wh) = (1,13,13,4,2)

    ### adjust confidence
    pred_box_conf = sigmoid.(y_pred[:,:,:,:,5])# prob bb
    #size(pred_box_conf) = (1,13,13,4)

    ### adjust class probabilities
    pred_box_class = y_pred[:,:,:,:,6:end] # prC1, prC2, ..., prC20
    #size(pred_box_class) = (1,13,13,4,20)

    return pred_box_xy,pred_box_wh,pred_box_conf,pred_box_class
end

function extract_ground_truth(y_true::Array{Float32,5})
    # size(y_true) = (1,13,13,4,25)
    true_box_xy = y_true[:,:,:,:,1:2]
    # size(true_box_xy) = (1,13,13,4,2)
    true_box_wh = y_true[:,:,:,:,3:4]
    # size(true_box_wh) = (1,13,13,4,2)
    true_box_conf = y_true[:,:,:,:,5]
    # size(true_box_conf) = (1,13,13,4)
    true_box_class = dropdims(argmax(y_true[:,:,:,:,6:end], dims=5);dims=5)#?????
    # size(true_box_class) = (1,13,13,4)??
    return true_box_xy, true_box_wh, true_box_conf, true_box_class
end

"""
```
coord_mask:      np.array of shape (Nbatch, Ngrid h, N grid w, N anchor, 1)
lambda_{coord} L_{i,j}^{obj}
```
"""
function calc_loss_xywh(true_box_conf,LAMBDA_COORD,true_box_xy, pred_box_xy,true_box_wh,pred_box_wh)
    # lambda_{coord} L_{i,j}^{obj}
    # np.array of shape (Nbatch, Ngrid h, N grid w, N anchor, 1)
    coord_mask   = expand_dims(true_box_conf, -1) .* LAMBDA_COORD
    # size(coord_mask) = (1,13,13,4,1)
    nb_coord_box = sum(Float32.(coord_mask .> 0.0))
    #size(nb_coord_box) = 1
    loss_xy      = sum((true_box_xy.-pred_box_xy).^2 .* coord_mask) / (nb_coord_box + 1e-6) / 2.0
    loss_wh      = sum((true_box_wh.-pred_box_wh).^2 .* coord_mask) / (nb_coord_box + 1e-6) / 2.0
    return loss_xy + loss_wh, coord_mask
end

"""
```
== input ==
true_box_conf  : tensor of shape (N batch, N grid h, N grid w, N anchor)
true_box_class : tensor of shape (N batch, N grid h, N grid w, N anchor), containing class index
pred_box_class : tensor of shape (N batch, N grid h, N grid w, N anchor, N class)
LAMBDA_CLASS    : 1.0
== output ==
class_mask
if object exists in this (grid_cell, anchor) pair and the class object receive nonzero weight
class_mask[iframe,igridy,igridx,ianchor] = 1
else:
0
```
"""
function calc_loss_class(true_box_conf,LAMBDA_CLASS, true_box_class,pred_box_class)
    class_mask   = true_box_conf  .* LAMBDA_CLASS
    # size(class_mask) = (1,13,13,4)
    nb_class_box = sum(Float32.(class_mask .> 0.0))
    loss_class   = sparse_softmax_cross_entropy_with_logits(true_box_class, pred_box_class)#????tied to true_box_class argmax issue
    loss_class   = sum(loss_class .* class_mask) / (nb_class_box + 1e-6)
    return loss_class
end


"""
== INPUT ==
true_box_xy,pred_box_xy, true_box_wh and pred_box_wh must have the same shape length

p1 : pred_mins = (px1,py1)
p2 : pred_maxs = (px2,py2)
t1 : true_mins = (tx1,ty1)
t2 : true_maxs = (tx2,ty2)
             p1______________________
             |      t1___________   |
             |       |           |  |
             |_______|___________|__|p2
                     |           |rmax
                     |___________|
                                  t2
intersect_mins : rmin = t1  = (tx1,ty1)
intersect_maxs : rmax = (rmaxx,rmaxy)
intersect_wh   : (rmaxx - tx1, rmaxy - ty1)
"""
function get_intersect_area(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh)

    true_box_wh_half = true_box_wh ./ 2.0
    true_mins    = true_box_xy .- true_box_wh_half
    true_maxes   = true_box_xy .+ true_box_wh_half

    pred_box_wh_half = pred_box_wh ./ 2.0
    pred_mins    = pred_box_xy .- pred_box_wh_half
    pred_maxes   = pred_box_xy .+ pred_box_wh_half

    intersect_mins  = max.(pred_mins,  true_mins)
    intersect_maxes = min.(pred_maxes, true_maxes)
    intersect_wh    = max.(intersect_maxes .- intersect_mins, 0.0)
    if ndims(true_box_xy) == 5
        intersect_areas = intersect_wh[:,:,:,:, 1] .* intersect_wh[:,:,:,:, 2]
        true_areas = true_box_wh[:,:,:,:, 1] .* true_box_wh[:,:,:,:, 2]
        pred_areas = pred_box_wh[:,:,:,:, 1] .* pred_box_wh[:,:,:,:, 2]
    elseif ndims(true_box_xy) == 6
        intersect_areas = intersect_wh[:,:,:,:,:, 1] .* intersect_wh[:,:,:,:,:, 2]
        true_areas = true_box_wh[:,:,:,:,:, 1] .* true_box_wh[:,:,:,:,:, 2]
        pred_areas = pred_box_wh[:,:,:,:,:, 1] .* pred_box_wh[:,:,:,:,:, 2]
    end
    union_areas = pred_areas .+ true_areas .- intersect_areas
    iou_scores  = intersect_areas ./ union_areas
    return iou_scores
end

"""
== input ==

true_box_conf : tensor of shape (N batch, N grid h, N grid w, N anchor )
true_box_xy   : tensor of shape (N batch, N grid h, N grid w, N anchor , 2)
true_box_wh   : tensor of shape (N batch, N grid h, N grid w, N anchor , 2)
pred_box_xy   : tensor of shape (N batch, N grid h, N grid w, N anchor , 2)
pred_box_wh   : tensor of shape (N batch, N grid h, N grid w, N anchor , 2)

== output ==

true_box_conf : tensor of shape (N batch, N grid h, N grid w, N anchor)

true_box_conf value depends on the predicted values
true_box_conf = IOU_{true,pred} if objecte exist in this anchor else 0
"""
function calc_IOU_pred_true_assigned(true_box_conf,true_box_xy, true_box_wh,pred_box_xy,  pred_box_wh)
    iou_scores = get_intersect_area(true_box_xy,true_box_wh,pred_box_xy,pred_box_wh)
    true_box_conf_IOU = iou_scores .* true_box_conf
    return true_box_conf_IOU
end

"""
```
== input ==
pred_box_xy : tensor of shape (N batch, N grid h, N grid w, N anchor, 2)
pred_box_wh : tensor of shape (N batch, N grid h, N grid w, N anchor, 2)
true_boxes  : tensor of shape (N batch, N grid h, N grid w, N anchor, 2)
== output ==
best_ious
for each iframe,
best_ious[iframe,igridy,igridx,ianchor] contains
the IOU of the object that is most likely included (or best fitted)
within the bounded box recorded in (grid_cell, anchor) pair
NOTE: a same object may be contained in multiple (grid_cell, anchor) pair
from best_ious, you cannot tell how may actual objects are captured as the "best" object
```
"""
function calc_IOU_pred_true_best(pred_box_xy,pred_box_wh,true_boxes)
    true_box_xy = true_boxes[:,:,:,:,:, 1:2]           # (N batch, 1, 1, 1, TRUE_BOX_BUFFER, 2)
    true_box_wh = true_boxes[:,:,:,:,:, 3:4]           # (N batch, 1, 1, 1, TRUE_BOX_BUFFER, 2)

    pred_box_xy = expand_dims(pred_box_xy, 5) # (N batch, N grid_h, N grid_w, N anchor, 1, 2)
    pred_box_wh = expand_dims(pred_box_wh, 5) # (N batch, N grid_h, N grid_w, N anchor, 1, 2)

    iou_scores  =  get_intersect_area(true_box_xy,true_box_wh,pred_box_xy,pred_box_wh) # (N batch, N grid_h, N grid_w, N anchor, 50)

    best_ious = dropdims(maximum(iou_scores, dims=5);dims=5 ) # (N batch, N grid_h, N grid_w, N anchor)
    return best_ious
end

"""
```
== input ==
best_ious           : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
true_box_conf       : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
true_box_conf_IOU   : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
LAMBDA_NO_OBJECT    : 1.0
LAMBDA_OBJECT       : 5.0
== output ==
conf_mask : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
conf_mask[iframe, igridy, igridx, ianchor] = 0
when there is no object assigned in (grid cell, anchor) pair and the region seems useless i.e.
y_true[iframe,igridx,igridy,4] = 0 "and" the predicted region has no object that has IoU > 0.6
conf_mask[iframe, igridy, igridx, ianchor] =  NO_OBJECT_SCALE
when there is no object assigned in (grid cell, anchor) pair but region seems to include some object
y_true[iframe,igridx,igridy,4] = 0 "and" the predicted region has some object that has IoU > 0.6
conf_mask[iframe, igridy, igridx, ianchor] =  OBJECT_SCALE
when there is an object in (grid cell, anchor) pair
```
"""
function get_conf_mask(best_ious, true_box_conf, true_box_conf_IOU,LAMBDA_NO_OBJECT, LAMBDA_OBJECT)
    conf_mask = Float32.(best_ious .< 0.6) .* (1 .- true_box_conf) .* LAMBDA_NO_OBJECT
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask .+ true_box_conf_IOU .* LAMBDA_OBJECT
    return conf_mask
end

"""
```
== input ==
conf_mask         : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
true_box_conf_IOU : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
pred_box_conf     : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
```
"""
function calc_loss_conf(conf_mask,true_box_conf_IOU, pred_box_conf)
    # the number of (grid cell, anchor) pair that has an assigned object or
    # that has no assigned object but some objects may be in bounding box.
    # N conf
    nb_conf_box  = sum(Float32.(conf_mask .> 0.0))
    loss_conf    = sum((true_box_conf_IOU .- pred_box_conf).^2 .* conf_mask)  / (nb_conf_box  + 1e-6) / 2.0
    return loss_conf
end

"""
```
y_true : (N batch, N grid h, N grid w, N anchor, 4 + 1 + N classes)
y_true[irow, i_gridh, i_gridw, i_anchor, :4] = center_x, center_y, w, h
center_x : The x coordinate center of the bounding box.
Rescaled to range between 0 and N gird  w (e.g., ranging between [0,13)
center_y : The y coordinate center of the bounding box.
Rescaled to range between 0 and N gird  h (e.g., ranging between [0,13)
w        : The width of the bounding box.
Rescaled to range between 0 and N gird  w (e.g., ranging between [0,13)
h        : The height of the bounding box.
Rescaled to range between 0 and N gird  h (e.g., ranging between [0,13)
y_true[irow, i_gridh, i_gridw, i_anchor, 4] = ground truth confidence
ground truth confidence is 1 if object exists in this (anchor box, gird cell) pair
y_true[irow, i_gridh, i_gridw, i_anchor, 5 + iclass] = 1 if the object is in category <iclass> else 0
=====================================================
tensor that connect to the YOLO model's hack input
=====================================================
true_boxes
=========================================
training parameters specification example
=========================================
gridsize           = (13,13)
batchsize          = 34
anchors = np.array([1.07709888,  1.78171903,  # anchor box 1, width , height
2.71054693,  5.12469308,  # anchor box 2, width,  height
10.47181473, 10.09646365,  # anchor box 3, width,  height
5.48531347,  8.11011331]) # anchor box 4, width,  height
λno_object = 1.0
λobject    = 5.0
λcoord     = 1.0
λclass     = 1.0
```
"""
function custom_loss_core(y_true,y_pred,true_boxes,GRIDSIZE::Tuple{Int,Int},
        BATCH_SIZE::Int,ANCHORS,LAMBDA_COORD,LAMBDA_CLASS,LAMBDA_NO_OBJECT, LAMBDA_OBJECT)
    nboxes = Int64(length(ANCHORS)/2)
    # Step 1: Adjust prediction output
    cell_grid   = get_cell_grid(GRIDSIZE,BATCH_SIZE,nboxes)
    pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = adjust_scale_prediction(y_pred,cell_grid,ANCHORS)
    # Step 2: Extract ground truth output
    true_box_xy, true_box_wh, true_box_conf, true_box_class = extract_ground_truth(y_true::Array{Float32,5})
    # Step 3: Calculate loss for the bounding box parameters
    loss_xywh, coord_mask = calc_loss_xywh(true_box_conf,LAMBDA_COORD,true_box_xy, pred_box_xy,true_box_wh,pred_box_wh)
    # Step 4: Calculate loss for the class probabilities
    loss_class  = calc_loss_class(true_box_conf,LAMBDA_CLASS,true_box_class,pred_box_class)
    # Step 5: For each (grid cell, anchor) pair,
    #         calculate the IoU between predicted and ground truth bounding box
    true_box_conf_IOU =
        calc_IOU_pred_true_assigned(true_box_conf,true_box_xy, true_box_wh,pred_box_xy,  pred_box_wh)
    # Step 6: For each predicted bounded box from (grid cell, anchor box),
    #         calculate the best IOU, regardless of the ground truth anchor box that each object gets assigned.
    best_ious = calc_IOU_pred_true_best(pred_box_xy,pred_box_wh,true_boxes)
    # Step 7: For each grid cell, calculate the L_{i,j}^{noobj}
    conf_mask = get_conf_mask(best_ious, true_box_conf, true_box_conf_IOU,LAMBDA_NO_OBJECT, LAMBDA_OBJECT)
    # Step 8: Calculate loss for the confidence
    loss_conf = calc_loss_conf(conf_mask,true_box_conf_IOU, pred_box_conf)

    loss = loss_xywh + loss_conf + loss_class
    return loss
end
