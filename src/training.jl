using Knet:Param,adam
using IterTools

function train(sets::Settings; epochs = 25, lr = 1e-4, numberofimages = 500)
    model = YOLO.v2_tiny.load(sets)
    YOLO.loadWeights!(model, sets)
    ##make last layer random and param
    model.layers[16].w = xtype(randn(Float32,1, 1, 1024, sets.cell_bboxes * (5 + sets.num_classes))/(sets.grid_x*sets.grid_y))
    model.layers[16].b = xtype(randn(Float32,1,1,sets.cell_bboxes * (5 + sets.num_classes),1)/(sets.grid_x*sets.grid_y))
    model.layers[16].w = Param(model.layers[16].w)
    model.layers[16].b = Param(model.layers[16].b)
    voc = YOLO.datasets.VOC.populate()
    vocloaded = YOLO.load(voc, sets, indexes = Vector(1:numberofimages)) ## create according to image number
    y_batch,b_batch= prepbatches(vocloaded.labels,sets)
    y_batch = reshape(y_batch,13*13*5*25,:)
    b_batch = reshape(b_batch,50*4,:)
    total_batch = vcat(b_batch,y_batch)
    #minibatch and training
    dtrn = minibatch(vocloaded.imstack_mat,total_batch,sets.minibatch_size; xtype = xtype, ytype = xtype, shuffle=true,partial = true)
    optimizer = adam(model,ncycle(dtrn,epochs);lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)
    progress!(optimizer)
    return model
end





#Create Y_batch and b_batch
function prepbatches(labels::Vector{Vector{TruthLabel}},sets::Settings)
    total = Array{Array{Float32,4},1}()
    btotal = Array{Array{Float32,5},1}()
    for i in 1:length(labels)
        onedim = zeros(Float32,13,13,5,25)
        onedimb = zeros(Float32,1,1,1,50,4)
        for k in 1:length(labels[i])
            rate =  sets.grid_x
            x = labels[i][k].bbox.x * rate
            y = labels[i][k].bbox.y * rate
            w = labels[i][k].bbox.w * rate
            h = labels[i][k].bbox.h * rate
            classNo = labels[i][k].class
            cx = Int32(floor(x+w/2)) + 1
            cy = Int32(floor(y+h/2)) + 1
            fillLocation!(onedim,x,y,w,h,classNo,cx,cy,sets)
            onedimb[1,1,1,k,1] = x + w/2
            onedimb[1,1,1,k,2] = y + h/2
            onedimb[1,1,1,k,3] = w
            onedimb[1,1,1,k,4] = h
        end
        push!(total,onedim)
        push!(btotal,onedimb)
    end
    return cat(total...,dims=5),cat(btotal...,dims=6)
end

function fillLocation!(arr,x,y,w,h,classNo,cx,cy,settings)
    ious = Array{Float32,1}()
    for i in 1:length(settings.anchors) # Find best iou match and fill only this part of array
        res = ioumatch(0,0,settings.anchors[i][1],settings.anchors[i][2],0,0,w,h)
        push!(ious,res)
    end
    loc = argmax(ious)
     #Fill this location
     while arr[cy,cx,loc,5] == 1
         ious[loc] = 0.0
         loc = argmax(ious)
     end
     arr[cy,cx,loc,1] = x + w/2
     arr[cy,cx,loc,2] = y + h/2
     arr[cy,cx,loc,3] = w
     arr[cy,cx,loc,4] = h
     arr[cy,cx,loc,5] = 1
     arr[cy,cx,loc,classNo + 5] = 1

end
