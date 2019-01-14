# YOLO.jl
# loss.jl

# Basic helper functions

```
sigmoid(x) = 1 / (1 + exp(-x))
```
sigmoid(x) = 1 / (1 + exp(-x))

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


```
(BATCH_SIZE, GRID_H, GRID_W, BOX, 2)
Helper function to assure that the bounding box x and y are in the grid cell scale

```
function get_cell_grid(GRID_W,GRID_H,BATCH_SIZE,BOX)
    cell_grid = Array{Float64}(undef,BATCH_SIZE,GRID_H,GRID_W,BOX,2)
    for batch in 1:BATCH_SIZE
        for gh in 1:GRID_H
            for gw in 1:GRID_W
                for box in 1:BOX
                    cell_grid[batch,gh,gw,box,1] =  gw
                    cell_grid[batch,gh,gw,box,2] =  gh
                end
            end
        end
    end
    return cell_grid
end

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
function adjust_scale_prediction(y_pred, cell_grid, ANCHORS)  

    BOX = int(len(ANCHORS)/2)
    ## cell_grid is of the shape of 

    ### adjust x and y  
    # the bounding box bx and by are rescaled to range between 0 and 1 for given gird.
    # Since there are BOX x BOX grids, we rescale each bx and by to range between 0 to BOX + 1
    pred_box_xy = sigmoid.(y_pred[..., :2]) + cell_grid # bx, by

    ### adjust w and h
    # exp to make width and height positive
    # rescale each grid to make some anchor "good" at representing certain shape of bounding box 
    pred_box_wh = exp.(y_pred[..., 2:4]) * np.reshape(ANCHORS,[1,1,1,BOX,2]) # bw, bh

    ### adjust confidence 
    pred_box_conf = sigmoid.(y_pred[..., 4])# prob bb

    ### adjust class probabilities 
    pred_box_class = y_pred[..., 5:] # prC1, prC2, ..., prC20

    return pred_box_xy,pred_box_wh,pred_box_conf,pred_box_class
end

function extract_ground_truth(y_true)    
    true_box_xy    = y_true[..., 0:2] # bounding box x, y coordinate in grid cell scale 
    true_box_wh    = y_true[..., 2:4] # number of cells accross, horizontally and vertically
    true_box_conf  = y_true[...,4]    # confidence 
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    return true_box_xy, true_box_wh, true_box_conf, true_box_class
end

```
coord_mask:      np.array of shape (Nbatch, Ngrid h, N grid w, N anchor, 1)
lambda_{coord} L_{i,j}^{obj}     

```
function calc_loss_xywh(true_box_conf,COORD_SCALE,true_box_xy, pred_box_xy,true_box_wh,pred_box_wh)  


    # lambda_{coord} L_{i,j}^{obj} 
    # np.array of shape (Nbatch, Ngrid h, N grid w, N anchor, 1)
    coord_mask  = expand_dims(true_box_conf, axis=-1) * COORD_SCALE 
    nb_coord_box = sum(float32.(coord_mask > 0.0))
    loss_xy      = sum((true_box_xy-pred_box_xy).^2 * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh      = sum((true_box_wh-pred_box_wh).^2 * coord_mask) / (nb_coord_box + 1e-6) / 2.
    return loss_xy + loss_wh, coord_mask 
end

```
== input ==    
true_box_conf  : tensor of shape (N batch, N grid h, N grid w, N anchor)
true_box_class : tensor of shape (N batch, N grid h, N grid w, N anchor), containing class index
pred_box_class : tensor of shape (N batch, N grid h, N grid w, N anchor, N class)
CLASS_SCALE    : 1.0

== output ==  
class_mask
if object exists in this (grid_cell, anchor) pair and the class object receive nonzero weight
class_mask[iframe,igridy,igridx,ianchor] = 1 
else: 
0 
```
function calc_loss_class(true_box_conf,CLASS_SCALE, true_box_class,pred_box_class)

    class_mask   = true_box_conf  * CLASS_SCALE ## L_{i,j}^obj * lambda_class

    nb_class_box = sum(float32.(class_mask > 0.0))
    loss_class   = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = true_box_class, 
        logits = pred_box_class)
    loss_class   = sum(loss_class * class_mask) / (nb_class_box + 1e-6)   
    return loss_class 
end


"""
bbox_iou(bbox_true::Array{Float64,1}, bbox_pred::Array{Float64,1}; xywh=true)

Returns the Intersection of Union (IoU) of two bounding boxes 
IoU = Area of Overlap / Area of Union
"""
function bbox_iou(bbox_true::Array{Float64,1}, bbox_pred::Array{Float64,1}; xywh=true)

    if xywh
        # Transform from center and width to exact coordinates
        bt_x1, bt_x2 = bbox_true[1] - bbox_true[3] / 2, bbox_true[1] + bbox_true[3] / 2
        bt_y1, bt_y2 = bbox_true[2] - bbox_true[4] / 2, bbox_true[2] + bbox_true[4] / 2
        bp_x1, bp_x2 = bbox_pred[1] - bbox_pred[3] / 2, bbox_pred[1] + bbox_pred[3] / 2
        bp_y1, bp_y2 = bbox_pred[2] - bbox_pred[4] / 2, bbox_pred[2] + bbox_pred[4] / 2
    else
        # Get the coordinates of bounding boxes
        bt_x1, bt_y1, bt_x2, bt_y2 = bbox_true[1], bbox_true[2], bbox_true[3], bbox_true[4]
        bp_x1, bp_y1, bp_x2, bp_y2 = bbox_pred[1], bbox_pred[2], bbox_pred[3], bbox_pred[4]
    end
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = maximum([bt_x1, bp_x1])
    inter_rect_y1 = maximum([bt_y1, bp_y1])
    inter_rect_x2 = minimum([bt_x2, bp_x2])
    inter_rect_y2 = minimum([bt_y2, bp_y2])
    # Intersection area
    inter_area = clamp(inter_rect_x2 - inter_rect_x1 + 1, 0, Inf) * clamp(
        inter_rect_y2 - inter_rect_y1 + 1, 0, Inf)
    # Union Area
    bt_area = (bt_x2 - bt_x1 + 1) * (bt_y2 - bt_y1 + 1)
    bp_area = (bp_x2 - bp_x1 + 1) * (bp_y2 - bp_y1 + 1)

    iou = inter_area / (bt_area + bp_area - inter_area + 1e-16)

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
function bbox_iou(bboxes_true::Array{Array{Float64,1},1},bboxes_pred::Array{Array{Float64,1},1}; xywh=true)
    n = size(bboxes_true,1)
    m = size(bboxes_pred,1)
    ious = zeros(Float64,n,m)
    for i = 1:n
        for j  = 1:m
            ious[i,j] = bbox_iou(bboxes_true[i],bboxes_pred[j],xywh=xywh)
        end
    end

    return ious
end


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
function calc_IOU_pred_true_best(pred_box_xy,pred_box_wh,true_boxes)   

    true_xy = true_boxes[..., 0:2]           # (N batch, 1, 1, 1, TRUE_BOX_BUFFER, 2)
    true_wh = true_boxes[..., 2:4]           # (N batch, 1, 1, 1, TRUE_BOX_BUFFER, 2)

    pred_xy = expand_dims(pred_box_xy, 4) # (N batch, N grid_h, N grid_w, N anchor, 1, 2)
    pred_wh = expand_dims(pred_box_wh, 4) # (N batch, N grid_h, N grid_w, N anchor, 1, 2)

    iou_scores  =  get_intersect_area(true_xy,
        true_wh,
        pred_xy,
        pred_wh) # (N batch, N grid_h, N grid_w, N anchor, 50)   

    best_ious = maximum(iou_scores, dims=4) # (N batch, N grid_h, N grid_w, N anchor)
    return best_ious
end

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
function get_conf_mask(best_ious, true_box_conf, true_box_conf_IOU,LAMBDA_NO_OBJECT, LAMBDA_OBJECT)    


    conf_mask = float32.(best_ious < 0.6) * (1 - true_box_conf) * LAMBDA_NO_OBJECT
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + true_box_conf_IOU * LAMBDA_OBJECT
    return conf_mask 
end

```
== input ==

conf_mask         : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
true_box_conf_IOU : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
pred_box_conf     : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
```

function calc_loss_conf(conf_mask,true_box_conf_IOU, pred_box_conf)  

    # the number of (grid cell, anchor) pair that has an assigned object or
    # that has no assigned object but some objects may be in bounding box.
    # N conf
    nb_conf_box  = sum(float32.(conf_mask  > 0.0))
    loss_conf    = sum((true_box_conf_IOU-pred_box_conf) * conf_mask).^2  / (nb_conf_box  + 1e-6) / 2.
    return loss_conf
end


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
GRID_W             = 13
GRID_H             = 13
BATCH_SIZE         = 34
ANCHORS = np.array([1.07709888,  1.78171903,  # anchor box 1, width , height
2.71054693,  5.12469308,  # anchor box 2, width,  height
10.47181473, 10.09646365,  # anchor box 3, width,  height
5.48531347,  8.11011331]) # anchor box 4, width,  height
LAMBDA_NO_OBJECT = 1.0
LAMBDA_OBJECT    = 5.0
LAMBDA_COORD     = 1.0
LAMBDA_CLASS     = 1.0
```
function custom_loss_core(y_true,y_pred,true_boxes,GRID_W,GRID_H,BATCH_SIZE,ANCHORS,LAMBDA_COORD,LAMBDA_CLASS,LAMBDA_NO_OBJECT, LAMBDA_OBJECT)

    BOX = int(len(ANCHORS)/2)    
    # Step 1: Adjust prediction output
    cell_grid   = get_cell_grid(GRID_W,GRID_H,BATCH_SIZE,BOX)
    pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = adjust_scale_prediction(y_pred,cell_grid,ANCHORS)
    # Step 2: Extract ground truth output
    true_box_xy, true_box_wh, true_box_conf, true_box_class = extract_ground_truth(y_true)
    # Step 3: Calculate loss for the bounding box parameters
    loss_xywh, coord_mask = calc_loss_xywh(true_box_conf,LAMBDA_COORD,
        true_box_xy, pred_box_xy,true_box_wh,pred_box_wh)
    # Step 4: Calculate loss for the class probabilities
    loss_class  = calc_loss_class(true_box_conf,LAMBDA_CLASS,
        true_box_class,pred_box_class)
    # Step 5: For each (grid cell, anchor) pair, 
    #         calculate the IoU between predicted and ground truth bounding box
    true_box_conf_IOU = true_box_conf * bbox_iou(bbox_true,bbox_pred)
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

function custom_loss(y_true, y_pred)
    loss = custom_loss_core(y_true,y_pred,true_boxes,GRID_W,GRID_H,BATCH_SIZE,ANCHORS,LAMBDA_COORD,LAMBDA_CLASS,LAMBDA_NO_OBJECT, LAMBDA_OBJECT)

    return loss
end
