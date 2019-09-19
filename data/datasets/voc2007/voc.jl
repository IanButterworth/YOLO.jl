module VOC

import ...BBOX, ...TruthLabel, ...LabelledImageDataset


import LightXML: content,
                 get_elements_by_tagname,
                 parse_file,
                 find_element,
                 root


include(joinpath(@__DIR__, "..", "..", "..", "src", "common.jl"))

const namesdic = Dict("aeroplane"=>1,"bicycle"=>2,"bird"=>3, "boat"=>4,
            "bottle"=>5,"bus"=>6,"car"=>7,"cat"=>8,"chair"=>9,
            "cow"=>10,"diningtable"=>11,"dog"=>12,"horse"=>13,"motorbike"=>14,
            "person"=>15,"pottedplant"=>16,"sheep"=>17,"sofa"=>18,"train"=>19,"tvmonitor"=>20)

labels_dir = joinpath(
    datasets_dir,
    "voc2007",
    "VOCdevkit",
    "VOC2007",
    "Annotations",
)
images_dir = joinpath(datasets_dir, "voc2007", "VOCdevkit", "VOC2007", "JPEGImages")

"""
    populate()

Populate VOC dataset as a `LabelledImageDataset`.
"""
function populate()
    if !isdir(joinpath(@__DIR__, "VOCdevkit", "VOC2007"))
        @error """VOC dataset is not downloaded. Download with `YOLO.download_dataset("voc2007")` and try again."""
        return nothing
    end
    @info "Populating VOC dataset"
    image_paths, label_paths = collectTruthLabelImagePairs(labels_dir, images_dir)
    labels = loadTruthLabel.(label_paths)
    objectcounts = countobjects(labels)

    return LabelledImageDataset(
        name = "VOC",
        objects = flipdict(namesdic),
        objectcounts = objectcounts,
        image_size_lims = (500, 500),
        images_dir = images_dir,
        labels_dir = labels_dir,
        image_paths = image_paths,
        label_paths = label_paths,
        labels = labels,
    )
end

"""
    countobjects(labels::Vector{Vector{TruthLabel}})

Collect objects and count object occurances across entire dataset
"""
function countobjects(labels::Vector{Vector{TruthLabel}})
    occurances = Int[]
    for label in labels
        append!(occurances, map(x -> x.class, label))
    end
    objects = sort(unique(occurances))
    objectcounts = map(x -> sum(occurances .== x), objects)
    return objectcounts
end

"""
    collectTruthLabelImagePairs(labels_dir::String, images_dir::String)

Collects all labels and images' directories, checks for matched label and image files.
"""
function collectTruthLabelImagePairs(labels_dir::String, images_dir::String)
    labels, images = String[], String[]
    excluded = 0
    for (root, dirs, files) in walkdir(labels_dir)
        filter!(x -> endswith(x,".xml") && !startswith(x,"."), files)
        label_filepaths = joinpath.(labels_dir, files)
        image_filepaths = joinpath.(
            images_dir,
            map(x -> string(first(split(x, ".xml")), ".jpg"), files),
        )
        image_exists = isfile.(image_filepaths)
        append!(labels, label_filepaths[image_exists])
        append!(images, image_filepaths[image_exists])
        excluded += sum(.!image_exists) #Count missing images.
    end
    (excluded > 0) && @info "$(excluded) label files excluded due to missing image files"
    return images, labels
end

"""
    loadTruthLabel(label_file::String)

Loads label file into `TruthLabel` struct.
"""
function loadTruthLabel(label_file::String)
    xdoc = parse_file(label_file)
    xroot = root(xdoc)
    ces = get_elements_by_tagname(xroot, "size")
    image_width = parse(Int32, content(find_element(ces[1], "width")))
    image_height = parse(Int32, content(find_element(ces[1], "height")))
    ces = get_elements_by_tagname(xroot, "object")
    YLs = TruthLabel[]
    for i = 1:length(ces)
        name = content(find_element(ces[i], "name"))
        difficult = parse(Int32, content(find_element(ces[i], "difficult")))
        if difficult == 0 #TODO allow different difficulties
            bbox_loc = find_element(ces[i], "bndbox")
            xmin = parse(Int32, content(find_element(bbox_loc, "xmin"))) /
                   image_width
            xmax = parse(Int32, content(find_element(bbox_loc, "xmax"))) /
                   image_width
            ymin = parse(Int32, content(find_element(bbox_loc, "ymin"))) /
                   image_height
            ymax = parse(Int32, content(find_element(bbox_loc, "ymax"))) /
                   image_height
            class = get(namesdic,name,-1)
            class == -1 && @error "Class name mismatch found : $(name) isn't in class list"
            push!(
                YLs,
                TruthLabel(
                    bbox = BBOX(
                        x = xmin,
                        y = ymin,
                        w = xmax - xmin,
                        h = ymax - ymin,
                    ),
                    class = class,
                    difficult = difficult,
                ),
            )
        end
    end
    return YLs
end
end #module
