function ioumatch(x1,y1,w1,h1,x2,y2,w2,h2)
        r1 = x1 + w1
        l1 = x1
        t1 = y1
        b1 = y1 + h1
        r2 = x2 + w2
        l2 = x2
        t2 = y2
        b2 = y2 + h2
        a = min(r1,r2)
        b = max(t1,t2)
        c = max(l1,l2)
        d = min(b1,b2)
        intersec = (d-b)*(a-c)
        return intersec/(w1*h1+w2*h2-intersec)
end


function getcellgrid(GRID_W,GRID_H,BATCH_SIZE,BOX)
    cell_x = Array{Float32,5}((reshape(repeat(Vector(1:GRID_W),GRID_H),GRID_H, GRID_W, 1, 1, 1)))
    cell_y = permutedims(cell_x, [2,1,3,4,5])
    cell_grid = repeat(cat(cell_y,cell_x; dims=4),1,1,BOX,1,BATCH_SIZE)
    cell_grid .= cell_grid.-1
    return typearr(cell_grid)
end

function adjust_predictions(pred,cell_grid,ANCHORS)
    pred = reshape(pred,13,13,25,5,:)
    pred = permutedims(pred,[1,2,4,3,5])
    BOX =  5
    pred_box_xy = sigmoid.(pred[:,:,:,1:2,:]) .+ cell_grid
    pred_box_wh = exp.(pred[:,:,:,3:4,:]) .* permutedims(reshape(ANCHORS,1,1,2,5,1),[1,2,4,3,5])
    pred_box_conf = sigmoid.(pred[:,:,:,5:5,:])
    pred_box_class = pred[:,:,:,6:end,:]
    return pred_box_xy,pred_box_wh,pred_box_conf,pred_box_class
end

function extractgroundtruth(y_true)
    true_box_xy    = y_true[:,:,:,1:2,:] # bounding box x, y coordinate in grid cell scale
    true_box_wh    = y_true[:,:,:,3:4,:] # number of cells accross, horizontally and vertically
    true_box_conf  = y_true[:,:,:,5:5,:] # confidence
    return true_box_xy, true_box_wh, true_box_conf,  y_true[:,:,:,6:end,:]
end

function calc_loss_xywh(true_box_conf,COORD_SCALE,true_box_xy, pred_box_xy,true_box_wh,pred_box_wh)
    coord_mask = true_box_conf .* COORD_SCALE #13,13,5,1,64lük array. 0lar ve 1*COORD_SCALEden oluşuyor
    nb_coord_box = sum(coord_mask .> 0)
    loss_xy = sum(square.(true_box_xy-pred_box_xy) .* coord_mask) / (nb_coord_box + 1e-6) / 2.0
    loss_wh = sum(square.(true_box_wh-pred_box_wh) .* coord_mask) / (nb_coord_box + 1e-6) / 2.0
    return loss_xy + loss_wh, coord_mask
end

function calc_loss_class(true_box_conf, CLASS_SCALE, pred_box_class, pred_all_class)
    class_mask   = true_box_conf  .* CLASS_SCALE
    nb_class_box = sum((class_mask .> 0))
    loss_class = sum(-log.(softmax(pred_box_class,dims=4)).*pred_all_class, dims = 4)
    loss_class = sum(loss_class .* class_mask) ./ (nb_class_box + 1e-6)
    return loss_class
end

function get_intersect_area(true_xy,true_wh,pred_xy,pred_wh)
    true_wh_half = true_wh ./ Float32(2.0)
    true_mins = true_xy .- true_wh_half
    true_maxes = true_xy .+ true_wh_half

    pred_wh_half = pred_wh ./ Float32(2.0)
    pred_mins = pred_xy .- pred_wh_half
    pred_maxes = pred_xy .+ pred_wh_half

    intersect_mins = max.(pred_mins,true_mins)
    intersect_maxes = min.(pred_maxes,true_maxes)
    intersect_wh = max.(intersect_maxes-intersect_mins,Float32(0.0))

    if ndims(pred_xy) == 5
        intersect_areas = intersect_wh[:,:,:,1:1,:] .* intersect_wh[:,:,:,2:2,:]
        true_areas = true_wh[:,:,:, 1:1,:] .* true_wh[:,:,:,2:2,:]
        pred_areas = pred_wh[:,:,:, 1:1,:] .* pred_wh[:,:,:,2:2,:]
    else
        intersect_areas = intersect_wh[:,:,:,:,1,:] .* intersect_wh[:,:,:,:,2,:]
        true_areas = true_wh[:,:,:,:,1,:] .* true_wh[:,:,:,:,2,:]
        pred_areas = pred_wh[:,:,:, :,1,:] .* pred_wh[:,:,:,:,2,:]

    end
    union_areas = pred_areas .+ true_areas .- intersect_areas
    iou_scores =  intersect_areas ./ union_areas
    return iou_scores
end

function iou_assigned(true_box_conf, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh)
    iou_scores = get_intersect_area(true_box_xy,true_box_wh,pred_box_xy,pred_box_wh)
    true_IOU = iou_scores .* true_box_conf
    return true_IOU
end

function iou_best(pred_box_xy,pred_box_wh,b_batch)
    true_xy = b_batch[:,:,:,:,1:2,:]
    true_wh = b_batch[:,:,:,:,3:4,:]
    pred_xy = reshape(pred_box_xy, 13,13,5,1,2,:)
    pred_wh = reshape(pred_box_xy, 13,13,5,1,2,:)
    iou_scores = get_intersect_area(true_xy,true_wh,pred_xy,pred_wh)
    best_ious = maximum(iou_scores; dims=4)
    return best_ious
end

function get_conf_mask(best_ious, true_box_conf, true_box_conf_IOU,LAMBDA_NO_OBJECT, LAMBDA_OBJECT)
    if GPU
        conf_mask = (best_ious .< 0.6) .* (1 .- true_box_conf) .* LAMBDA_NO_OBJECT
    else
        conf_mask = Float32.(best_ious .< 0.6) .* (1 .- true_box_conf) .* LAMBDA_NO_OBJECT
    end
    conf_mask .= conf_mask .+ true_box_conf_IOU .* LAMBDA_OBJECT
    return conf_mask
end

function loss_conf(conf_mask,true_box_conf_IOU,pred_box_conf)
    nb_conf_box = sum(conf_mask .> 0.0)
    loss_conf = sum(square.(true_box_conf_IOU - pred_box_conf) .* conf_mask) / (nb_conf_box + 1e-6) / 2 #Float32.(best_ious .< 0.6)
    return loss_conf
end

function yololoss(total_batch, y_pred)

    y_true = total_batch[201:end,:]
    y_true = reshape(y_true,13,13,5,25,:)
    b_batch = total_batch[1:200,:]
    b_batch = reshape(b_batch,1,1,1,50,4,:)

    #adjust prediction
    cell = getcellgrid(13,13,size(y_true)[5],5)
    pred_box_xy,pred_box_wh,pred_box_conf,pred_box_class = adjust_predictions(y_pred,cell,ANCHORS)
    #ground truth
    true_box_xy, true_box_wh, true_box_conf,  pred_all_class = extractgroundtruth(y_true)
    #Calculate xy and wh loss
    loss_xywh, coord_mask = calc_loss_xywh(true_box_conf,coord_scale, true_box_xy,pred_box_xy,true_box_wh,pred_box_wh)
    #Calculate class loss
    loss_class = calc_loss_class(true_box_conf,class_scale,pred_box_class,pred_all_class)
    #Find IOU between assigned and ground truth
    assign_conf = iou_assigned(true_box_conf, true_box_xy, true_box_wh, Knet.value(pred_box_xy), Knet.value(pred_box_wh))
    #Find best iou_scores
    best_ious = iou_best(Knet.value(pred_box_xy), Knet.value(pred_box_wh),b_batch)
    #Create conf_mask to calculate confidence loss
    conf_mask = get_conf_mask(best_ious,true_box_conf,assign_conf,noobject_scale,object_scale)
    #Calculate conf loss
    conf_loss = loss_conf(conf_mask,assign_conf,pred_box_conf)

    total_loss = loss_xywh + conf_loss + loss_class
    return total_loss
end
