
include(joinpath(models_dir, "v2_tiny", "v2_tiny.jl"))
import .v2_tiny: v2TinyChain, loadWeights!

include(joinpath(models_dir, "v2", "v2.jl"))
import .v2: v2Chain, loadWeights!
