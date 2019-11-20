module YOLOV2_608

import ...flipdict, ...Settings

namesdic = COCO.namesdic


function load(;minibatch_size=1)
    return Settings(
        dataset_description = "COCO",
        source = "https://pjreddie.com/media/files/yolov2-voc.weights",
        weights_filepath = joinpath(@__DIR__,"v2_voc.weights"),
        image_shape = (608, 608),
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
