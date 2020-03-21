import ..GPU


import ..YOLO: loadWeights!

export loadWeights!

"""
    flipkernel(x)

Flips given kernel
"""
flipkernel(x) = x[end:-1:1, end:-1:1, :, :]

"""
    updateconv!(c, gamma, mean, variance)

Applies batch-normalization to given convolutional layer
"""
function updateconv!(c, gamma, mean, variance)
    gamma4 = reshape(gamma, 1, 1, 1, :)
    variance4 = reshape(variance, 1, 1, 1, :)
    gamma3 = reshape(gamma, 1, 1, :, 1)
    mean3 = reshape(mean, 1, 1, :, 1)
    variance3 = reshape(variance, 1, 1, :, 1)
    #update c.w
    a = convert(Float32, 0.001)
    c.w = (c.w .* gamma4) ./ sqrt.(variance4 .+ a)
    #update c.b
    c.b = c.b .- (gamma3 .* mean3 ./ sqrt.(variance3 .+ a))
    return c.w, c.b
end

"""
    loadWeights!(model::YOLO.v2_tiny.v2Chain, weights_filepath::String)

Loads layers' weights from given weights file path.
"""
function loadWeights!(model::Chain_layers, settings::Settings, nr_constants = 4)
    outlength = settings.cell_bboxes * (5 + settings.num_classes)

    io = open(settings.weights_filepath, "r") #do io
        readconstants!(io, nr_constants)
        #First Conv layer
        loadconv!(model.layers[1].layers[1], io, 3, 3, 3, 32)#25
        #Second Conv layer
        loadconv!(model.layers[1].layers[3], io, 3, 3, 32, 64)#37
        #Third Conv layer
        loadconv!(model.layers[1].layers[5], io, 3, 3, 64, 128)#49
        #4th Conv layer
        loadconv!(model.layers[1].layers[6], io, 1, 1, 128, 64)#57
        #5th Conv layer
        loadconv!(model.layers[1].layers[7], io, 3, 3, 64, 128)#65
        #6th Conv layer
        loadconv!(model.layers[1].layers[9], io, 3, 3, 128, 256)#77
	#7th Conv layer
        loadconv!(model.layers[1].layers[10], io, 1, 1, 256, 128)#85
	#8th Conv layer
        loadconv!(model.layers[1].layers[11], io, 3, 3, 128, 256)#93
	#9th Conv layer
        loadconv!(model.layers[1].layers[13], io, 3, 3, 256, 512)#105
	#10th Conv layer
        loadconv!(model.layers[1].layers[14], io, 1, 1, 512, 256)#113
	#11th Conv layer
        loadconv!(model.layers[1].layers[15], io, 3, 3, 256, 512)#121
	#12th Conv layer
        loadconv!(model.layers[1].layers[16], io, 1, 1, 512, 256)#129
	#13th Conv layer
        loadconv!(model.layers[1].layers[17], io, 3, 3, 256, 512)#137
	#14th Conv layer
        loadconv!(model.layers[2].layers[2], io, 3, 3, 512, 1024)#149
	#15th Conv layer
        loadconv!(model.layers[2].layers[3], io, 1, 1, 1024, 512)#157
	#16th Conv layer
        loadconv!(model.layers[2].layers[4], io, 3, 3, 512, 1024)#165
	#17th Conv layer
        loadconv!(model.layers[2].layers[5], io, 1, 1, 1024, 512)#173
	#18th Conv layer
        loadconv!(model.layers[2].layers[6], io, 3, 3, 512, 1024)#181
	#19th Conv layer
        loadconv!(model.layers[2].layers[7], io, 3, 3, 1024, 1024)#192
	#20th Conv layer
        loadconv!(model.layers[2].layers[8], io, 3, 3, 1024, 1024)#200
	#21st Conv layer
        loadconv!(model.layers[3].layers[1], io, 1, 1, 512, 64)#211
	#22nd Conv layer
        loadconv!(model.layers[4].layers[1], io, 3, 3, 1280, 1024)#225
	#23rd Conv layer
        loadconv_last!(model.layers[4].layers[2], io, 1, 1, 1024, 125)#233 no batch norm here
        if GPU >= 0
          end
    # end
end

"""
    loadconv!(c, io, d1, d2, d3, d4)

Loads the io to given convolutional layer and updates it by batch-normalization
"""
function loadconv!(c, io::IOStream, d1, d2, d3, d4)
    read!(io, c.b)
    gamma = Array{Float32}(undef, d4)
    mean = Array{Float32}(undef, d4)
    variance = Array{Float32}(undef, d4)
    read!(io, gamma)
    read!(io, mean)
    read!(io, variance)
    toRead = Array{Float32}(undef, d4 * d3 * d2 * d1)
    read!(io, toRead)
    toRead = reshape(toRead, d1, d2, d3, d4)
    c.w = permutedims(toRead, [2, 1, 3, 4])
    c.w, c.b = updateconv!(c, gamma, mean, variance)
    c.w = flipkernel(c.w)
    if GPU >= 0
        c.w = KnetArray(c.w)
        c.b = KnetArray(c.b)
    end
end

# to be simplified. the weights file does not have batch norm data for the last layer
function loadconv_last!(c, io::IOStream, d1, d2, d3, d4)
    read!(io, c.b)
    toRead = Array{Float32}(undef, d4 * d3 * d2 * d1)
    read!(io, toRead)
    toRead = reshape(toRead, d1, d2, d3, d4)
    c.w = permutedims(toRead, [2, 1, 3, 4])
    c.w = flipkernel(c.w)
    if GPU >= 0
        c.w = KnetArray(c.w)
        c.b = KnetArray(c.b)
    end
end
"""
    readconstants!(io)

Read constant and unnecessary numbers from the io stream
"""
function readconstants!(io::IOStream, nr_constants = 4)
    # major = read(io, Int32)
    # minor = read(io, Int32)
    # revision = read(io, Int32)
    # iseen = read(io, Int32)
	# dummy = read(io, Float32)
	for i = 1:nr_constants
		read(io, Float32)
	end
end

## initial
# io = open(settings.weights_filepath, "r") #do io
# 	readconstants!(io)
# 	#First Conv layer
# 	loadconv!(model.layers[1], io, 3, 3, 3, 32)#25
# 	#Second Conv layer
# 	loadconv!(model.layers[3], io, 3, 3, 32, 64)#37
# 	#Third Conv layer
# 	loadconv!(model.layers[5], io, 3, 3, 64, 128)#49
# 	#4th Conv layer
# 	loadconv!(model.layers[6], io, 1, 1, 128, 64)#57
# 	#5th Conv layer
# 	loadconv!(model.layers[7], io, 3, 3, 64, 128)#65
# 	#6th Conv layer
# 	loadconv!(model.layers[9], io, 3, 3, 128, 256)#77
# 	#7th Conv layer
# 	loadconv!(model.layers[10], io, 1, 1, 256, 128)#85
# 	#8th Conv layer
# 	loadconv!(model.layers[11], io, 3, 3, 128, 256)#93
# 	#9th Conv layer
# 	loadconv!(model.layers[13], io, 3, 3, 256, 512)#105
# 	#10th Conv layer
# 	loadconv!(model.layers[14], io, 1, 1, 512, 256)#113
# 	#11th Conv layer
# 	loadconv!(model.layers[15], io, 3, 3, 256, 512)#121
# 	#12th Conv layer
# 	loadconv!(model.layers[16], io, 1, 1, 512, 256)#129
# 	#13th Conv layer
# 	loadconv!(model.layers[17], io, 3, 3, 256, 512)#137
# 	#14th Conv layer
# 	loadconv!(model.layers[19], io, 3, 3, 512, 1024)#149
# 	#15th Conv layer
# 	loadconv!(model.layers[20], io, 1, 1, 1024, 512)#157
# 	#16th Conv layer
# 	loadconv!(model.layers[21], io, 3, 3, 512, 1024)#165
# 	#17th Conv layer
# 	loadconv!(model.layers[22], io, 1, 1, 1024, 512)#173
# 	#18th Conv layer
# 	loadconv!(model.layers[23], io, 3, 3, 512, 1024)#181
# 	#19th Conv layer
# 	loadconv!(model.layers[24], io, 3, 3, 1024, 1024)#192
# 	#20th Conv layer
# 	loadconv!(model.layers[25], io, 3, 3, 1024, 1024)#200
# 	#21st Conv layer
# 	loadconv!(model.layers[26], io, 1, 1, 512, 64)#211
# 	#22nd Conv layer
# 	loadconv!(model.layers[27], io, 3, 3, 1280, 1024)#225
# 	#23rd Conv layer
# 	loadconv_last!(model.layers[28], io, 1, 1, 1024, 125)#233
# 	if GPU >= 0
# 	  end
