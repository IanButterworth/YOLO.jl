#Author: Yavuz Faruk Bakman
#Date: 15/08/2019

#Flips given kernel
flipkernel(x) = x[end:-1:1, end:-1:1, :, :]

#Applies batch-normalization to given convolutional layer
function updateconv!(c,gama,mean,varriance)
gama4 = reshape(gama,1,1,1,:)
varriance4 = reshape(varriance,1,1,1,:)
gama3 = reshape(gama,1,1,:,1)
mean3 = reshape(mean,1,1,:,1)
varriance3 = reshape(varriance,1,1,:,1)
#update c.w
a = convert(Float32,0.001)
c.w = (c.w .* gama4) ./ sqrt.(varriance4 .+ a)
#update c.b
c.b = c.b .- (gama3 .* mean3 ./ sqrt.(varriance3 .+ a))
return c.w ,c.b
end

#loads layers' weights from given file
function getweights(model, file)
    println("Loading weights")
    readconstants!(f)
    #First Conv layer
    loadconv!(model.layers[1],file,3,3,3,16)
    #Second Conv layer
    loadconv!(model.layers[3],file,3,3,16,32)
    #Third Conv layer
    loadconv!(model.layers[5],file,3,3,32,64)
    #4th Conv layer
    loadconv!(model.layers[7],file,3,3,64,128)
    #5th Conv layer
    loadconv!(model.layers[9],file,3,3,128,256)
    #6th Conv layer
    loadconv!(model.layers[11],file,3,3,256,512)
    #YoloPad
    model.layers[13].w[1,1,1,1] = 1
    #7th Conv layer
    loadconv!(model.layers[14],file,3,3,512,1024)
    #8th Conv layer
    loadconv!(model.layers[15],file,3,3,1024,1024)
    #last layer
    read!(file, model.layers[16].b)
    toRead = Array{Float32}(UndefInitializer(), 1024*125);
    read!(file, toRead)
    toRead = reshape(toRead,1,1,1024,125)
    model.layers[16].w = permutedims(toRead,[2,1,3,4])
    model.layers[16].w = flipkernel(model.layers[16].w)

    if gpu() >= 0
        model.layers[16].w = KnetArray(model.layers[16].w)
        model.layers[16].b = KnetArray(model.layers[16].b)
        model.layers[13].w = KnetArray(model.layers[13].w)
    end
    println("Weights loaded")
end

#loads the file to given convolutional layer and updates it by batch-normalization
function loadconv!(c,file,d1,d2,d3,d4)
    read!(file, c.b)
    gama= Array{Float32}(UndefInitializer(), d4);
    mean = Array{Float32}(UndefInitializer(), d4);
    variance = Array{Float32}(UndefInitializer(), d4);
    read!(file,gama)
    read!(file,mean)
    read!(file,variance)
    toRead = Array{Float32}(UndefInitializer(), d4*d3*d2*d1);
    read!(file,toRead)
    toRead = reshape(toRead,d1,d2,d3,d4)
    c.w = permutedims(toRead,[2,1,3,4])
    c.w,c.b = updateconv!(c,gama,mean,variance)
    c.w = flipkernel(c.w)
    if gpu() >= 0
        c.w = KnetArray(c.w)
        c.b = KnetArray(c.b)
    end
end

#read constant and unnecessary numbers from the file
function readconstants!(file)
    major  = read(f,Int32)
    minor = read(f,Int32)
    revision = read(f,Int32)
    iseen = read(f,Int32)
end
