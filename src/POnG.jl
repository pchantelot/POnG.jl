module POnG

    # Load Julia's implementation of DMSUITE
    using LinearAlgebra, ToeplitzMatrices
    include(joinpath(@__DIR__,"Differentiation/chebpoints.jl"))
    include(joinpath(@__DIR__,"Differentiation/chebdif.jl"))
    include(joinpath(@__DIR__,"Differentiation/lpoly.jl"))
    include(joinpath(@__DIR__,"Differentiation/lgrpointsleft.jl"))
    include(joinpath(@__DIR__,"Differentiation/lgrdiffleft.jl"))
    export chebdif, cheb1extrema, lgrpointsleft, lgrdiffleft

    # Load Elastic tensor
    include(joinpath(@__DIR__,"Material/createC.jl"))
    export C_elastic, C_MR, C_GT, C_C, C_MRSH, C_MRSH2, C_GTSH, C_GTSH2, C_CSH, C_GMR, C_GG, C_GC, C_DCMR, C_DCMR2, C_DCGT
    # Layer coupling
    include(joinpath(@__DIR__,"Material/assemblelayer.jl"))
    export assemblelayer

    # Load discretization
    using Kronecker
    include(joinpath(@__DIR__,"Elastodynamics/elastodynamics.jl"))
    export  elastodynamics1D, elastodynamics1D_tube, elastodynamics2D, Celastodynamics2D

    # Load NonlinearEigenproblems to use polyeig
    using NonlinearEigenproblems
    export polyeig, PEP

    # Make 1D and 2D discretization class of results for use in processing functions.
    struct result1D
        udof::UnitRange{Int64}
        N::Int64
        ω::Array{Float64}
        k::Array{ComplexF64}
        u::Array{ComplexF64}
    end
    struct result2D
        udof::UnitRange{Int64}
        N::Int64
        P::Int64
        ω::Array{Float64}
        k::Array{ComplexF64}
        u::Array{ComplexF64}
    end
    export result1D, result2D

    # Postprocessing functions
    include(joinpath(@__DIR__,"Processing/SCMprocessing.jl"))
    export modedirection, modesymmetry

end
