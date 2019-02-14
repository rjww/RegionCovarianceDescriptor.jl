mutable struct SymmetricTensor{T} <: AbstractArray{T,4}
    w::Int
    h::Int
    f::Int
    contents::Array{T,3}

    function SymmetricTensor{T}(w::Int, h::Int, f::Int) where T
        contents = Array{T}(undef, w, h, div(f * (f + 1), 2))
        new{T}(w, h, f, contents)
    end
end

function Base.size(o::SymmetricTensor)
    return (o.w, o.h, o.f, o.f)
end

function Base.getindex(o::SymmetricTensor, i::Int, j::Int, k::Int, l::Int)
    o.contents[i, j, convert_indices(o.f, order_indices(k, l)...)]
end

function Base.setindex!(o::SymmetricTensor, x, i::Int, j::Int, k::Int, l::Int)
    o.contents[i, j, convert_indices(o.f, order_indices(k, l)...)] = x
end

function order_indices(i::Int, j::Int)
    return j < i ? (j, i) : (i, j)
end

function convert_indices(n::Int, i::Int, j::Int)
    return n * (i - 1) + j - div(i * (i - 1), 2)
end

"""
```
descriptor = RegionCovariance(feature_img)
```
"""
struct RegionCovariance
    P::Array{Float64,3}
    Q::SymmetricTensor{Float64}
    nrows::Int
    ncols::Int
    nfeatures::Int

    function RegionCovariance(F::AbstractArray{Float64,3})
        nrows, ncols, nfeatures = length.(axes(F))
        P = featurewise_integral_images(F, nrows, ncols, nfeatures)
        Q = feature_product_integral_images(F, nrows, ncols, nfeatures)
        new(P, Q, nrows, ncols, nfeatures)
    end

    # Calculate the integral image of each feature layer of `F`. This is equivalent to
    # calling `integral_image(Fᵢ)` in a loop, where `Fᵢ` is a view of ith feature layer of
    # `F`, but reproducing the functionality here results in significantly fewer
    # allocations.
    function featurewise_integral_images(F::AbstractArray{Float64,3},
                                         nrows::Int, ncols::Int, nfeatures::Int)
        P = Array{Float64,3}(undef, nrows, ncols, nfeatures)

        @inbounds for f in 1:nfeatures
            for r in 1:nrows, c in 1:ncols
                P[r,c,f] = F[r,c,f]
                P[r,c,f] += r > 1 ? P[r-1,c,f] : 0.0
                P[r,c,f] += c > 1 ? P[r,c-1,f] : 0.0
                P[r,c,f] -= r > 1 && c > 1 ? P[r-1,c-1,f] : 0.0
            end
        end

        return P
    end

    # Given a feature image with 𝑛 features, where each pixel is a corresponds to a feature
    # vector 𝐱 ∈ ℝⁿ, calculate the 𝑛 × 𝑛 matrix 𝐴 for each 𝐱 such that 𝐴[𝑖,𝑗] = 𝐱ᵢ ⋅ 𝐱ⱼ, and
    # then the integral image of that matrix. Note that 𝐴 in this case is symmetric: a
    # custom data structure is used to leverage this symmetry and save space.
    function feature_product_integral_images(F::AbstractArray{Float64,3},
                                             nrows::Int, ncols::Int, nfeatures::Int)
        Q = SymmetricTensor{Float64}(nrows, ncols, nfeatures)

        @inbounds for fᵢ in 1:nfeatures
            for fⱼ in fᵢ:nfeatures
                for r in 1:nrows, c in 1:ncols
                    Q[r,c,fᵢ,fⱼ] = F[r,c,fᵢ] * F[r,c,fⱼ]
                    Q[r,c,fᵢ,fⱼ] += r > 1 ? Q[r-1,c,fᵢ,fⱼ] : 0.0
                    Q[r,c,fᵢ,fⱼ] += c > 1 ? Q[r,c-1,fᵢ,fⱼ] : 0.0
                    Q[r,c,fᵢ,fⱼ] -= r > 1 && c > 1 ? Q[r-1,c-1,fᵢ,fⱼ] : 0.0
                end
            end
        end

        return Q
    end
end

function covariance_matrix(d::RegionCovariance, rows::AbstractRange, cols::AbstractRange)
    𝐂 = Array{Float64}(undef, d.nfeatures, d.nfeatures)
    return covariance_matrix!(𝐂, d, rows, cols)
end

function covariance_matrix!(𝐂::AbstractArray{Float64,2}, d::RegionCovariance,
                            rows::AbstractRange, cols::AbstractRange)
    row₀, col₀ = first.((rows, cols))
    row₁, col₁ = last.((rows, cols))
    return covariance_matrix!(𝐂, d, row₀, col₀, row₁, col₁)
end

function covariance_matrix(d::RegionCovariance,
                           row₀::Integer, col₀::Integer,
                           row₁::Integer, col₁::Integer)
    𝐂 = Array{Float64}(undef, d.nfeatures, d.nfeatures)
    return covariance_matrix!(𝐂, d, row₀, col₀, row₁, col₁)
end

function covariance_matrix!(𝐂::AbstractArray{Float64,2}, d::RegionCovariance,
                            row₀::Integer, col₀::Integer,
                            row₁::Integer, col₁::Integer)
    region_size = (row₁-row₀+1)*(col₁-col₀+1)

    𝐐 = sum_feature_products_over_region(d.Q, Val{d.nfeatures}(), row₀, col₀, row₁, col₁)
    𝐩 = sum_over_region_by_feature(d.P, Val{d.nfeatures}(), row₀, col₀, row₁, col₁)
    result = (𝐐 - ((𝐩 * 𝐩') / region_size)) / (region_size-1)

    𝐂 .= result
end

function sum_feature_products_over_region(Q::SymmetricTensor{Float64}, ::Val{nfeatures},
                                          row₀::Integer, col₀::Integer, row₁::Integer,
                                          col₁::Integer) where nfeatures
    𝐐 = zeros(MMatrix{nfeatures,nfeatures,Float64})

    @inbounds for fᵢ in 1:nfeatures
        for fⱼ in fᵢ:nfeatures
            𝐐[fᵢ,fⱼ] = Q[row₁,col₁,fᵢ,fⱼ]
            𝐐[fᵢ,fⱼ] -= row₀ > 1 ? Q[row₀-1,col₁,fᵢ,fⱼ] : 0.0
            𝐐[fᵢ,fⱼ] -= col₀ > 1 ? Q[row₁,col₀-1,fᵢ,fⱼ] : 0.0
            𝐐[fᵢ,fⱼ] += row₀ > 1 && col₀ > 1 ? Q[row₀-1,col₀-1,fᵢ,fⱼ] : 0.0
            𝐐[fⱼ,fᵢ] = 𝐐[fᵢ,fⱼ]
        end
    end

    return 𝐐
end

function sum_over_region_by_feature(P::AbstractArray{Float64,3}, ::Val{nfeatures},
                                    row₀::Integer, col₀::Integer, row₁::Integer,
                                    col₁::Integer) where nfeatures
    𝐩 = zeros(MVector{nfeatures,Float64})

    @inbounds for f in 1:nfeatures
        𝐩[f] = P[row₁,col₁,f]
        𝐩[f] -= row₀ > 1 ? P[row₀-1,col₁,f] : 0.0
        𝐩[f] -= col₀ > 1 ? P[row₁,col₀-1,f] : 0.0
        𝐩[f] += row₀ > 1 && col₀ > 1 ? P[row₀-1,col₀-1,f] : 0.0
    end

    return 𝐩
end
