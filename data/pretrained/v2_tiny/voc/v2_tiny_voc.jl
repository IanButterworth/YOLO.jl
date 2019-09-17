module v2_tiny_voc

import ...flipdict, ...Settings

#20 classes on Voc dataset
# 2 dictionaries to access number<->class by O(1)
const namesdic = Dict("aeroplane"=>1,"bicycle"=>2,"bird"=>3, "boat"=>4,
            "bottle"=>5,"bus"=>6,"car"=>7,"cat"=>8,"chair"=>9,
            "cow"=>10,"diningtable"=>11,"dog"=>12,"horse"=>13,"motorbike"=>14,
            "person"=>15,"pottedplant"=>16,"sheep"=>17,"sofa"=>18,"train"=>19,"tvmonitor"=>20)

#Yolo V2 pre-trained boxes width and height
const anchors = [(1.08,1.19),  (3.42,4.41),  (6.63,11.38),  (9.42,5.11),  (16.62,10.52)]

function load()
    return Settings(
        dataset_description = "VOC",
        source = "https://pjreddie.com/media/files/yolov2-tiny-voc.weights",
        weights_filepath = joinpath(@__DIR__,"v2_tiny_voc.weights"),
        image_shape = (416, 416),
        image_channels = 3,
        namesdic = namesdic,
        numsdic = flipdict(namesdic), #Flip the key value pair to help lookup by nums
        anchors = anchors,
        num_classes = length(namesdic)
    )
end

end #module
