module VOC

import ...Label, ...LabelledImageDataset

import LightXML: content,
                 get_elements_by_tagname,
                 parse_file,
                 find_element,
                 root

include(joinpath(@__DIR__, "..", "..", "..", "src", "common.jl"))

labels_dir = joinpath(
    datasets_dir,
    "voc",
    "VOCdevkit",
    "VOC2007",
    "Annotations",
)
images_dir = joinpath(datasets_dir, "voc", "VOCdevkit", "VOC2007", "JPEGImages")

"""
    populate()

Populate VOC dataset as a `LabelledImageDataset`.
"""
function populate()
    if !isdir(joinpath(@__DIR__, "VOCdevkit", "VOC2007"))
        @error """VOC dataset is not downloaded. Download with `YOLO.download_dataset("voc")` and try again."""
        return nothing
    end
    @info "Populating VOC dataset"
    image_paths, label_paths = collectLabelImagePairs(labels_dir, images_dir)
    labels = loadLabel.(label_paths)
    objects, objectcounts = countobjects(labels)

    return LabelledImageDataset(
        name = "VOC",
        objects = objects,
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
    countobjects(labels::Vector{Vector{Label}})

Collect objects and count object occurances across entire dataset
"""
function countobjects(labels::Vector{Vector{Label}})
    occurances = String[]
    for label in labels
        append!(occurances, map(x -> x.name, label))
    end
    objects = sort(unique(occurances))
    objectcounts = map(x -> sum(occurances .== x), objects)
    return objects, objectcounts
end

"""
    collectLabelImagePairs(labels_dir::String, images_dir::String)

Collects all labels and images' directories, checks for matched label and image files.
"""
function collectLabelImagePairs(labels_dir::String, images_dir::String)
    labels, images = String[], String[]
    excluded = 0
    for (root, dirs, files) in walkdir(labels_dir)
        label_filepaths = joinpath.(
            labels_dir,
            filter(x -> occursin(".xml", x), files),
        )
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
    loadLabel(label_file::String)

Loads label file into `Label` struct.
"""
function loadLabel(label_file::String)
    xdoc = parse_file(label_file)
    xroot = root(xdoc)
    ces = get_elements_by_tagname(xroot, "size")
    image_width = parse(Int32, content(find_element(ces[1], "width")))
    image_height = parse(Int32, content(find_element(ces[1], "height")))
    ces = get_elements_by_tagname(xroot, "object")
    YLs = Label[]
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
            push!(
                YLs,
                Label(
                    x = xmin,
                    y = ymin,
                    w = xmax - xmin,
                    h = ymax - ymin,
                    name = name,
                    difficult = difficult,
                ),
            )
        end
    end
    return YLs
end
end #module
