module v2_voc

import ...flipdict, ...Settings

#20 classes on Voc dataset
# 2 dictionaries to access number<->class by O(1)
const namesdic = Dict("aeroplane"=>1,"bicycle"=>2,"bird"=>3, "boat"=>4,
            "bottle"=>5,"bus"=>6,"car"=>7,"cat"=>8,"chair"=>9,
            "cow"=>10,"diningtable"=>11,"dog"=>12,"horse"=>13,"motorbike"=>14,
            "person"=>15,"pottedplant"=>16,"sheep"=>17,"sofa"=>18,"train"=>19,"tvmonitor"=>20)

#Yolo V2 pre-trained boxes width and height
const anchors = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)]

function load(;minibatch_size=1)
    return Settings(
        dataset_description = "VOC2007",
        source = "https://pjreddie.com/media/files/yolov2-voc.weights",
        weights_filepath = joinpath(@__DIR__,"v2_voc.weights"),
        image_shape = (416, 416),
        image_channels = 3,
        namesdic = namesdic,
        numsdic = flipdict(namesdic), #Flip the key value pair to help lookup by nums
        anchors = anchors,
        num_classes = length(namesdic),
        minibatch_size = minibatch_size,
        grid_x = 19,
        grid_y = 19,
        cell_bboxes = 5
    )
end

end #module
