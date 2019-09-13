#Author: Yavuz Faruk Bakman
#Date: 15/08/2019

#process the input and save into given directory
saveOut(model,data,confth,iouth,res,number; record = true, location = "Output") = (saveOut!(model,args,confth,iouth,res,number; record = record, location = location) for args in data)

function saveOut!(model,args,confth,iouth,res,number; record = true, location = "Output")
    out = model(args[1])
    out = postprocessing(out,confth,iouth)
    a = out
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
#confth => confidence score threshold. 0.3 is recommended
#iouth => intersection over union threshold. if 2 images overlap more than this threshold, one of them is removed
function saveoutput(model,data,confth,iouth; record = true, location = "Output")
    res = []
    number = [0]
    println("Processing Input and Saving...")
    progress!(saveOut(model,data,confth,iouth,res,number; record = record, location = location))
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
acc(model,data,confth,iouth,iou,predictions) =(acc!(model,args,confth,iouth,iou,predictions) for args in data)

function acc!(model,args,confth,iouth,iou,predictions)
    out = model(args[1])
    out = postprocessing(out,confth,iouth)
    check = zeros(length(args[2][1])-2)
    sort!(out,by = x-> x[6],rev=true)
    for k in 1:length(out)
        tp,loc = istrue(out[k],args[2][1][3:length(args[2][1])],check,iou)
        push!(predictions[numsdic[out[k][5]]],(tp,out[k][6]))
        if tp
            check[loc] = 1
        end
    end
end
#confth => confidence score threshold. 0.0 for calculating accuracy
#iouth => intersection over union threshold. if 2 images overlap more than this threshold, one of them is removed
#iou => intersection over union. True positive threshold
function accuracy(model,data,confth,iouth,iou)
    predictions = Dict("aeroplane"=>[],"bicycle"=>[],"bird"=>[], "boat"=>[],
                    "bottle"=>[],"bus"=>[],"car"=>[],"cat"=>[],"chair"=>[],
                    "cow"=>[],"diningtable"=>[],"dog"=>[],"horse"=>[],"motorbike"=>[],
                    "person"=>[],"pottedplant"=>[],"sheep"=>[],"sofa"=>[],"train"=>[],"tvmonitor"=>[])
    apdic= Dict("aeroplane"=>0.0,"bicycle"=>0.0,"bird"=>0.0, "boat"=>0.0,
                    "bottle"=>0.0,"bus"=>0.0,"car"=>0.0,"cat"=>0.0,"chair"=>0.0,
                    "cow"=>0.0,"diningtable"=>0.0,"dog"=>0.0,"horse"=>0.0,"motorbike"=>0.0,
                    "person"=>0.0,"pottedplant"=>0.0,"sheep"=>0.0,"sofa"=>0.0,"train"=>0.0,"tvmonitor"=>0.0)

    println("Calculating accuracy...")
    progress!(acc(model,data,confth,iouth,iou,predictions))
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
    im, img_size, img_originalsize, padding = loadprepareimage(file,(416,416))
    im_input = Array{Float32}(undef,416,416,3,1)
    im_input[:,:,:,1] = permutedims(collect(channelview(im)),[2,3,1]);
    if gpu() >= 0 im_input = KnetArray(im_input) end
    res = model(im_input)
    a = postprocessing(res,0.3,0.3)
    for i in 1:length(a)
        drawsquare(im,a[i][1],a[i][2],a[i][3],a[i][4],padding)
        FreeTypeAbstraction.renderstring!(im, string(numsdic[a[i][5]]), face, (14,14)  ,Int32(round(a[i][2]))-padding[2],Int32(round(a[i][1]))-padding[1],halign=:hleft,valign=:vtop,bcolor=eltype(im)(1.0,1.0,1.0),fcolor=eltype(im)(0,0,0)) #use `nothing` to make bcolor transparent
    end
    p1 = padding[1]
    p2 = padding[2]
    display(im[1:end-p2,1:end-p1])
    if record save("outexample.jpg",im[1:end-p2,1:end-p1]) end
end

#post processing function.
#Confidence score threshold to select correct predictions. Recommended : 0.3
#IoU threshold to remove unnecessary predictions: Recommended:0.3
function postprocessing(out,confth,iouth)
    out = Array{Float32,4}(out)
    result = []
    RATE = 32
    for cy in 1:13
        for cx in 1:13
            for b in 1:5
                channel = (b-1)*(numClass + 5)
                tx = out[cy,cx,channel+1,1]
                ty = out[cy,cx,channel+2,1]
                tw = out[cy,cx,channel+3,1]
                th = out[cy,cx,channel+4,1]
                tc = out[cy,cx,channel+5,1]
                x = (sigmoid(tx) + cx-1) * RATE
                y = (sigmoid(ty) + cy-1) * RATE
                w = exp(tw) * anchors[b][1] * RATE
                h = exp(th) * anchors[b][2] * RATE
                conf = sigmoid(tc)
                classScores = out[cy,cx,channel+6:channel+25,1]
                classScores = softmax(classScores)
                classNo = argmax(classScores)
                bestScore = classScores[classNo]
                classConfidenceScore = conf*bestScore
                if classConfidenceScore > confth
                     p = (max(0.0,x-w/2),max(0.0,y-h/2),min(w,416.0),min(h,416.0),classNo,classConfidenceScore)
                     push!(result,p)
                end
            end
        end
    end
    result = nonmaxsupression(result,iouth)
    return result
end

#It removes the predictions overlapping.
function nonmaxsupression(results,iouth)
    sort!(results, by = x ->x[6],rev=true)
    for i in 1:length(results)
        k = i+1
        while k <= length(results)
            if ioumatch(results[i][1],results[i][2],results[i][3],results[i][4],
                results[k][1],results[k][2],results[k][3],results[k][4]) > iouth && results[i][5] == results[k][5]
                deleteat!(results,k)
                k = k - 1
            end
            k = k+1
        end
    end
 return results
end

#It calculates IoU score (overlapping rate)
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
