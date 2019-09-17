#Author: Yavuz Faruk Bakman
#Date: 15/08/2019

import ..GPU

#process the input and save into given directory
saveOut(model,data,conf_thresh,iou_thresh,res,number; record = true, location = "Output") = (saveOut!(model,args,conf_thresh,iou_thresh,res,number; record = record, location = location) for args in data)

function saveOut!(model,args,conf_thresh,iou_thresh,res,number; record = true, location = "Output")
    yolomat = model(args[1])
    yolomat = postprocess(yolomat,conf_thresh,iou_thresh)
    a = yolomat
    push!(res,a)
    im = args[2][1]
    p2 = 416-length(axes(im)[1][1:end])
    p1 = 416-length(axes(im)[2][1:end])
    padding = (p1,p2)
    for i in 1:length(a)
        drawsquare(im,a[i][1],a[i][2],a[i][3],a[i][4],padding)
        FreeTypeAbstraction.renderstring!(im, string(numsdic[a[i][5]]), face, (14,14)  ,Int32(round(a[i][2]))-padding[2],Int32(round(a[i][1]))-padding[1],halign=:hleft,valign=:vtop,bcolor=eltype(im)(1.0,1.0,1.0),fcolor=eltype(im)(0,0,0)) #use `nothing` to make bcolor transparent
    end
    number[1] = number[1] + 1
    num = number[1]
    if record
        if !isdir(location)
            mkdir(location)
        end
        save(string(location,"/$num.jpg"),im[1:end-p2,1:end-p1])
    end
end
#conf_thresh => confidence score threshold. 0.3 is recommended
#iou_thresh => intersection over union threshold. if 2 images overlap more than this threshold, one of them is removed
function saveoutput(model,data,conf_thresh,iou_thresh; record = true, location = "Output")
    res = []
    number = [0]
    println("Processing Input and Saving...")
    progress!(saveOut(model,data,conf_thresh,iou_thresh,res,number; record = record, location = location))
    println("Saved all output")
    return res
end

#draw square to given image
function drawsquare(im,x,y,w,h,padding)
    x = Int32(round(x))-padding[1]
    y = Int32(round(y))-padding[2]
    w= Int32(round(w))
    h = Int32(round(h))

    draw!(im, LineSegment(Point(x,y), Point(x+w,y)))
    draw!(im, LineSegment(Point(x,y), Point(x,y+h)))
    draw!(im, LineSegment(Point(x+w,y), Point(x+w,y+h)))
    draw!(im, LineSegment(Point(x,y+h), Point(x+w,y+h)))
end

#Calculates accuracy for Voc Dataset
acc(model,data,conf_thresh,iou_thresh,iou,predictions) =(acc!(model,args,conf_thresh,iou_thresh,iou,predictions) for args in data)

function acc!(model,args,conf_thresh,iou_thresh,iou,predictions)
    yolomat = model(args[1])
    yolomat = postprocess(yolomat,conf_thresh,iou_thresh)
    check = zeros(length(args[2][1])-2)
    sort!(yolomat,by = x-> x[6],rev=true)
    for k in 1:length(yolomat)
        tp,loc = istrue(yolomat[k],args[2][1][3:length(args[2][1])],check,iou)
        push!(predictions[numsdic[yolomat[k][5]]],(tp,yolomat[k][6]))
        if tp
            check[loc] = 1
        end
    end
end
#conf_thresh => confidence score threshold. 0.0 for calculating accuracy
#iou_thresh => intersection over union threshold. if 2 images overlap more than this threshold, one of them is removed
#iou => intersection over union. True positive threshold
function accuracy(model,data,conf_thresh,iou_thresh,iou)
    predictions = Dict("aeroplane"=>[],"bicycle"=>[],"bird"=>[], "boat"=>[],
                    "bottle"=>[],"bus"=>[],"car"=>[],"cat"=>[],"chair"=>[],
                    "cow"=>[],"diningtable"=>[],"dog"=>[],"horse"=>[],"motorbike"=>[],
                    "person"=>[],"pottedplant"=>[],"sheep"=>[],"sofa"=>[],"train"=>[],"tvmonitor"=>[])
    apdic= Dict("aeroplane"=>0.0,"bicycle"=>0.0,"bird"=>0.0, "boat"=>0.0,
                    "bottle"=>0.0,"bus"=>0.0,"car"=>0.0,"cat"=>0.0,"chair"=>0.0,
                    "cow"=>0.0,"diningtable"=>0.0,"dog"=>0.0,"horse"=>0.0,"motorbike"=>0.0,
                    "person"=>0.0,"pottedplant"=>0.0,"sheep"=>0.0,"sofa"=>0.0,"train"=>0.0,"tvmonitor"=>0.0)

    println("Calculating accuracy...")
    progress!(acc(model,data,conf_thresh,iou_thresh,iou,predictions))
    for key in keys(predictions)
        sort!(predictions[key], by = x ->x[2],rev = true)
        tp = 0
        fp = 0
        total = totaldic[key]
        preRecall = []
        p = predictions[key]
        for i in 1:length(p)
            if p[i][1]
                tp = tp+1
            else
                fp = fp+1
            end
            if total==0
                push!(preRecall,[0,0])
            end
            push!(preRecall,[tp/(tp+fp),tp/total])
        end
        #smooth process
        rightMax = preRecall[length(preRecall)][1]
        location = length(preRecall)-1
        while(location >= 1)
            if preRecall[location][1] > rightMax
                rightMax = preRecall[location][1]
            else
                preRecall[location][1] = rightMax
            end
            location = location -1
        end
        #make calculation
        sum = 0
        for i in 2:length(preRecall)
            sum = sum + (preRecall[i][2]-preRecall[i-1][2]) * preRecall[i][1]
        end
            apdic[key] = sum
    end
    println("Calculated")
    return apdic
end

#Checks if given prediction is true positive or false negative
function istrue(prediction,labels,check,iou)
    min = iou
    result = false
    location = length(labels) + 1
    for i in 1:length(labels)
        if prediction[5] == namesdic[labels[i][5]] && check[i] == 0 && ioumatch(prediction[1],prediction[2],prediction[3],prediction[4],labels[i][1],labels[i][2],labels[i][3],labels[i][4]) > min
            min = ioumatch(prediction[1],prediction[2],prediction[3],prediction[4],labels[i][1],labels[i][2],labels[i][3],labels[i][4])
            result = true
            location = i
        end
    end
    return result ,location
end

function calculatemean(dict)
    sum = 0
    number = 0
    for key in keys(dict)
        sum = sum + dict[key]
        number = number + 1
    end
    return sum/number
end

#Displays an image's output on IDE
function displaytest(file,model; record = false)
    im, padding = loadResizePadImageToFit(file,(416,416))
    im_input = Array{Float32}(undef,416,416,3,1)
    im_input[:,:,:,1] = permutedims(collect(channelview(im)),[2,3,1]);
    if GPU >= 0 im_input = KnetArray(im_input) end
    res = model(im_input)
    a = postprocess(res,0.3,0.3)
    for i in 1:length(a)
        drawsquare(im,a[i][1],a[i][2],a[i][3],a[i][4],padding)
        FreeTypeAbstraction.renderstring!(im, string(numsdic[a[i][5]]), face, (14,14)  ,Int32(round(a[i][2]))-padding[2],Int32(round(a[i][1]))-padding[1],halign=:hleft,valign=:vtop,bcolor=eltype(im)(1.0,1.0,1.0),fcolor=eltype(im)(0,0,0)) #use `nothing` to make bcolor transparent
    end
    p1 = padding[1]
    p2 = padding[2]
    display(im[1:end-p2,1:end-p1])
    if record save("outexample.jpg",im[1:end-p2,1:end-p1]) end
end

"""
    postprocess(yolomat::Array{Float32}, conf_thresh::Float32, iou_thresh::Float32)

Post processing function.
Confidence score threshold to select correct predictions. Recommended : 0.3
IoU threshold to remove unnecessary predictions: Recommended:0.3
"""
function postprocess(yolomat::Array{Float32},settings::Settings; conf_thresh::T = 0.3, iou_thresh::T = 0.3) where {T<:AbstractFloat}
    detections = PredictLabel[]
    RATE = 32
    @views for cy in 1:13
        for cx in 1:13
            for b in 1:5
                channel = (b-1)*(settings.num_classes + 5)
                tx = yolomat[cy,cx,channel+1,1]
                ty = yolomat[cy,cx,channel+2,1]
                tw = yolomat[cy,cx,channel+3,1]
                th = yolomat[cy,cx,channel+4,1]
                tc = yolomat[cy,cx,channel+5,1]
                x = (sigmoid(tx) + cx-1) * RATE
                y = (sigmoid(ty) + cy-1) * RATE
                w = exp(tw) * settings.anchors[b][1] * RATE
                h = exp(th) * settings.anchors[b][2] * RATE

                conf = sigmoid(tc)
                classScores = yolomat[cy,cx,channel+6:channel+25,1]
                classScores = softmax(classScores)
                classNo = argmax(classScores)
                bestScore = classScores[classNo]
                classConfidenceScore = conf*bestScore
                if classConfidenceScore > conf_thresh
                    bbox = BBOX(max(0.0,x-w/2),max(0.0,y-h/2),min(w,416.0),min(h,416.0))
                    p = PredictLabel(bbox = bbox, class = classNo, conf =classConfidenceScore)
                    push!(detections, p)
                end
            end
        end
    end
    nonMaxSupression!(detections, iou_thresh)
    return detections
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
