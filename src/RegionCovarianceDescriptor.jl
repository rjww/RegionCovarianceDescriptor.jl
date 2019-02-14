module RegionCovarianceDescriptor

using StaticArrays

include("descriptor.jl")

export RegionCovariance,
       covariance_matrix,
       covariance_matrix!

end # module
