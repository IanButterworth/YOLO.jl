module pretrained
import ..pretrained_dir
include(joinpath(pretrained_dir, "v2_tiny", "voc2007", "v2_tiny_voc.jl"))
include(joinpath(pretrained_dir, "v2", "voc2007", "v2_voc.jl"))
end #module
