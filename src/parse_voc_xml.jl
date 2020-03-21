using EzXML

# struct for storing xml data
mutable struct Voc_xml
	folder::Array{String}
	filename::Array{String}
	width::Array{Int64}
	height::Array{Int64}
	depth::Array{Int64}
	segmented::Array{Int64}
	name::Array{String}
	pose::Array{String}
	truncated::Array{Int64}
	difficult::Array{Int64}
	xmin::Array{Int64}
	ymin::Array{Int64}
	xmax::Array{Int64}
	ymax::Array{Int64}
	Voc_xml() = new([],[],[],[],[],[],[],[],[],[],[],[],[],[])
end

function get_xml_meta(children)
	folder 	  = children[1].content
	filename  = children[2].content
	width 	  = elements(children[5])[1].content
	height 	  = elements(children[5])[1].content
	depth 	  = elements(children[5])[3].content
	segmented = children[6].content
	return (folder, filename, width, height, depth, segmented)
end

function upd_xml_data!(voc_xml_data, voc_xml_meta, children)
	for i = 7:length(children) # first 6 fields are meta
		push!(voc_xml_data.folder, voc_xml_meta[1])
		push!(voc_xml_data.filename, voc_xml_meta[2])
		push!(voc_xml_data.width, parse(Int64,voc_xml_meta[3]))
		push!(voc_xml_data.height, parse(Int64,voc_xml_meta[4]))
		push!(voc_xml_data.depth, parse(Int64,voc_xml_meta[5]))
		push!(voc_xml_data.segmented, parse(Int64,voc_xml_meta[6]))
		push!(voc_xml_data.name, elements(children[i])[1].content)
		push!(voc_xml_data.pose, elements(children[i])[2].content)
		push!(voc_xml_data.truncated, parse(Int64,elements(children[i])[3].content))
		push!(voc_xml_data.difficult, parse(Int64,elements(children[i])[4].content))
		if elements(children[i])[5].name == "bndbox"# skip all the person: head, hand etc....
			push!(voc_xml_data.xmin, parse(Int64,elements(elements(children[i])[5])[1].content))
			push!(voc_xml_data.ymin, parse(Int64,elements(elements(children[i])[5])[2].content))
			push!(voc_xml_data.xmax, parse(Int64,elements(elements(children[i])[5])[3].content))
			push!(voc_xml_data.ymax, parse(Int64,elements(elements(children[i])[5])[4].content))
		end
	end
	return nothing
end

function upd_xml_data!(voc_xml_data, voc_xml_path)
	for (root, dirs, files) in walkdir(voc_xml_path)
		for file in files
			xml_raw = String(read(voc_xml_path*"\\"*file))
			# Parse an XML string
			doc = parsexml(xml_raw)
			# Get the root element from `doc`.
			rt = doc.root
			children  = elements(rt)
			voc_xml_meta = get_xml_meta(children)
			upd_xml_data!(voc_xml_data, voc_xml_meta, children)
		end
	end
return nothing
end

voc_xml_path = raw"C:\Projects\TrafficSign\DataSet\VOCdevkit\VOC2007\Annotations";
# init empty struct
voc_xml_data = Voc_xml()
@time upd_xml_data!(voc_xml_data, voc_xml_path)
