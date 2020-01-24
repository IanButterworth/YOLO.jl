#Create Y_batch and b_batch
function prepbatches(out)
    total = Array{Array{Float32,4},1}()
    btotal = Array{Array{Float32,5},1}()
    # her seferinde hazırla sonra totale koy onu da catle return et
    for i in 1:length(out)
        onedim = zeros(Float32,13,13,5,25)
        onedimb = zeros(Float32,1,1,1,50,4)
        for k in 3:length(out[i])
            x = out[i][k][1] / 32   #Sanki 13*13 pixelmiş gibi davran
            y = out[i][k][2] / 32
            w = out[i][k][3] / 32
            h = out[i][k][4] / 32
            classNo = namesdic[out[i][k][5]]
            cx = Int32(floor(x+w/2)) + 1
            cy = Int32(floor(y+h/2)) + 1
            fillLocation!(onedim,x,y,w,h,classNo,cx,cy)
            onedimb[1,1,1,k-2,1] = x + w/2
            onedimb[1,1,1,k-2,2] = y + h/2
            onedimb[1,1,1,k-2,3] = w
            onedimb[1,1,1,k-2,4] = h
        end
        push!(total,onedim)
        push!(btotal,onedimb)
    end
    return cat(total...,dims=5),cat(btotal...,dims=6)
end

#Aynı yere birden fazla atama olabilir
function fillLocation!(arr,x,y,w,h,classNo,cx,cy)
    ious = Array{Float32,1}()
    for i in 1:length(anchors) # Find best iou match and fill only this part of array
        res = ioumatch(0,0,anchors[i][1],anchors[i][2],0,0,w,h)
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
