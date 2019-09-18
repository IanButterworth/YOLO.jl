import ..GPU

"""
    postprocess(yolomat::Array{Float32}, conf_thresh::Float32, iou_thresh::Float32)

Post processing function.
Confidence score threshold to select correct predictions. Recommended : 0.3
IoU threshold to remove unnecessary predictions: Recommended:0.3
"""
function postprocess(yolomat::Array{Float32},settings::Settings; conf_thresh::T = 0.3, iou_thresh::T = 0.3) where {T<:AbstractFloat}
    im_w = settings.image_shape[1]
    im_h = settings.image_shape[2]
    num_images = size(yolomat,4)
    all_detections = map(x->PredictLabel[],1:num_images)
    RATE = 32
    @views for i in 1:num_images
        for cy in 1:13
            for cx in 1:13
                for b in 1:5
                    channel = (b-1)*(settings.num_classes + 5)
                    tx = yolomat[cy,cx,channel+1,i]
                    ty = yolomat[cy,cx,channel+2,i]
                    tw = yolomat[cy,cx,channel+3,i]
                    th = yolomat[cy,cx,channel+4,i]
                    tc = yolomat[cy,cx,channel+5,i]
                    x = (sigmoid(tx) + cx-1) * RATE
                    y = (sigmoid(ty) + cy-1) * RATE
                    w = exp(tw) * (settings.anchors[b][1]) * RATE
                    h = exp(th) * (settings.anchors[b][2]) * RATE
                    conf = sigmoid(tc)
                    classScores = yolomat[cy,cx,channel+6:channel+25,i]
                    classScores = softmax(classScores)
                    classNo = argmax(classScores)
                    bestScore = classScores[classNo]
                    classConfidenceScore = conf*bestScore
                    if classConfidenceScore > conf_thresh
                        bbox = BBOX(
                            x = max(0.0,x-w/2)/im_w,
                            y = max(0.0,y-h/2)/im_h,
                            w = min(w,im_w)/im_w,
                            h = min(h,im_h)/im_h)
                        p = PredictLabel(bbox = bbox, class = classNo, conf = classConfidenceScore)
                        push!(all_detections[i], p)
                    end
                end
            end
        end
    end
    nonMaxSupression!.(all_detections, iou_thresh)
    if length(all_detections) == 1
        return all_detections[1]
    else
        return all_detections
    end
end

"""
    nonMaxSupression!(detections::Vector{PredictLabel}, iou_thresh::Float32)

Removes the predictions overlapping.
"""
function nonMaxSupression!(detections::Vector{PredictLabel}, iou_thresh::T) where {T<:AbstractFloat}
    sort!(detections, by = x -> x.conf, rev = true)
    for i = 1:length(detections)
        k = i + 1
        while k <= length(detections)
            iou = ioumatch(detections[i].bbox, detections[k].bbox)
            if iou > iou_thresh && (detections[i].class == detections[k].class)
                deleteat!(detections, k)
                k -= 1
            end
            k += 1
        end
    end
end

"""
    ioumatch(bbox1::BBOX, bbox2::BBOX)

Calculates IoU score (overlapping rate)
"""
function ioumatch(bbox1::BBOX, bbox2::BBOX)
    r1 = bbox1.x + bbox1.w
    l1 = bbox1.x
    t1 = bbox1.y
    b1 = bbox1.y + bbox1.h
    r2 = bbox2.x + bbox2.w
    l2 = bbox2.x
    t2 = bbox2.y
    b2 = bbox2.y + bbox2.h
    a = min(r1, r2)
    b = max(t1, t2)
    c = max(l1, l2)
    d = min(b1, b2)
    intersec = (d - b) * (a - c)
    return intersec / (bbox1.w * bbox1.h + bbox2.w * bbox2.h - intersec)
end


imgslice_to_rgbimg(imgslice) = collect(colorview(RGB, permutedims(imgslice,[3,1,2])))

"""
    renderResult(img::Array{Float32}, predictions::Vector{YOLO.PredictLabel}, settings::YOLO.Settings; save_file::String = "")

Display image with predicted bounding boxes overlaid. Optionally save render as a file (should end in .jpg, .png etc.)
"""
function renderResult(img::Array{Float32}, predictions::Vector{YOLO.PredictLabel}, settings::YOLO.Settings; save_file::String = "")
    cols = distinguishable_colors(settings.num_classes+1, [RGB(1,1,1)])[2:end]
    img_w = settings.image_shape[1]
    img_h = settings.image_shape[2]
    img_rgb = imgslice_to_rgbimg(img)
    scene = Scene(resolution = size(img_rgb') .* 2 )
    image!(scene, img_rgb, scale_plot=false, show_axis = false, limits = FRect(0, 0, img_h, img_w))
    rotate!(scene, -0.5pi)

    for p in predictions
        col = cols[p.class]
        rect = [p.bbox.y*img_h, p.bbox.x*img_w, p.bbox.h*img_h, p.bbox.w*img_w]
        poly!(scene, [Rectangle{Float32}(rect...)], color=RGBA(red(col),blue(col),green(col),clamp(p.conf*1.3,0.2,0.6)))
        name = get(settings.numsdic,p.class,"")
        conf_rounded = round(p.conf, digits=2)
        text!(scene, "$name\n$(conf_rounded)",
             position = (rect[1], rect[2]),
             align = (:left,  :top),
             colo = :black,
             textsize = 6,
             font = "Dejavu Sans"
         )
    end
    save_file != "" && Makie.save(save_file, scene)
    return scene
end

## TODO Convert the remainer

# #Calculates accuracy for Voc Dataset
# acc(model,data,conf_thresh,iou_thresh,iou,predictions) =(acc!(model,args,conf_thresh,iou_thresh,iou,predictions) for args in data)
#
# function acc!(model,args,conf_thresh,iou_thresh,iou,predictions)
#     yolomat = model(args[1])
#     yolomat = postprocess(yolomat,conf_thresh,iou_thresh)
#     check = zeros(length(args[2][1])-2)
#     sort!(yolomat,by = x-> x[6],rev=true)
#     for k in 1:length(yolomat)
#         tp,loc = istrue(yolomat[k],args[2][1][3:length(args[2][1])],check,iou)
#         push!(predictions[numsdic[yolomat[k][5]]],(tp,yolomat[k][6]))
#         if tp
#             check[loc] = 1
#         end
#     end
# end
# #conf_thresh => confidence score threshold. 0.0 for calculating accuracy
# #iou_thresh => intersection over union threshold. if 2 images overlap more than this threshold, one of them is removed
# #iou => intersection over union. True positive threshold
# function accuracy(model,data,conf_thresh,iou_thresh,iou)
#     predictions = Dict("aeroplane"=>[],"bicycle"=>[],"bird"=>[], "boat"=>[],
#                     "bottle"=>[],"bus"=>[],"car"=>[],"cat"=>[],"chair"=>[],
#                     "cow"=>[],"diningtable"=>[],"dog"=>[],"horse"=>[],"motorbike"=>[],
#                     "person"=>[],"pottedplant"=>[],"sheep"=>[],"sofa"=>[],"train"=>[],"tvmonitor"=>[])
#     apdic= Dict("aeroplane"=>0.0,"bicycle"=>0.0,"bird"=>0.0, "boat"=>0.0,
#                     "bottle"=>0.0,"bus"=>0.0,"car"=>0.0,"cat"=>0.0,"chair"=>0.0,
#                     "cow"=>0.0,"diningtable"=>0.0,"dog"=>0.0,"horse"=>0.0,"motorbike"=>0.0,
#                     "person"=>0.0,"pottedplant"=>0.0,"sheep"=>0.0,"sofa"=>0.0,"train"=>0.0,"tvmonitor"=>0.0)
#
#     println("Calculating accuracy...")
#     progress!(acc(model,data,conf_thresh,iou_thresh,iou,predictions))
#     for key in keys(predictions)
#         sort!(predictions[key], by = x ->x[2],rev = true)
#         tp = 0
#         fp = 0
#         total = totaldic[key]
#         preRecall = []
#         p = predictions[key]
#         for i in 1:length(p)
#             if p[i][1]
#                 tp = tp+1
#             else
#                 fp = fp+1
#             end
#             if total==0
#                 push!(preRecall,[0,0])
#             end
#             push!(preRecall,[tp/(tp+fp),tp/total])
#         end
#         #smooth process
#         rightMax = preRecall[length(preRecall)][1]
#         location = length(preRecall)-1
#         while(location >= 1)
#             if preRecall[location][1] > rightMax
#                 rightMax = preRecall[location][1]
#             else
#                 preRecall[location][1] = rightMax
#             end
#             location = location -1
#         end
#         #make calculation
#         sum = 0
#         for i in 2:length(preRecall)
#             sum = sum + (preRecall[i][2]-preRecall[i-1][2]) * preRecall[i][1]
#         end
#             apdic[key] = sum
#     end
#     println("Calculated")
#     return apdic
# end
#
# #Checks if given prediction is true positive or false negative
# function istrue(prediction,labels,check,iou)
#     min = iou
#     result = false
#     location = length(labels) + 1
#     for i in 1:length(labels)
#         if prediction[5] == namesdic[labels[i][5]] && check[i] == 0 && ioumatch(prediction[1],prediction[2],prediction[3],prediction[4],labels[i][1],labels[i][2],labels[i][3],labels[i][4]) > min
#             min = ioumatch(prediction[1],prediction[2],prediction[3],prediction[4],labels[i][1],labels[i][2],labels[i][3],labels[i][4])
#             result = true
#             location = i
#         end
#     end
#     return result ,location
# end
#
# function calculatemean(dict)
#     sum = 0
#     number = 0
#     for key in keys(dict)
#         sum = sum + dict[key]
#         number = number + 1
#     end
#     return sum/number
# end
