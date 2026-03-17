using POnG
using Test
using FileIO, JLD2
using LinearAlgebra, Kronecker

@testset verbose = true "POnG.jl" begin

    # 1. Check that all hyperelastic models default to the usual elastic tensor in the reference configuration without viscoelasticity. 
    # We use the usual ecoflex OO-30 parameters as an example and normalize by the shear modulus
    @testset "hyperelastic models" begin
        λ = 1.0 
        ρs = 1e3
        vL = 1e3
        μs = 23e3
        λs = ρs * vL^2 - 2*μs
        f = 0 # static comparison
        τ = 330e-6
        n = 0.3
        β = 0.3 

        # Small to moderate strain regime models
        α = 0.3 # balance coefficients C1 and C2
        @test real(C_MR(λ, λ, ρs, vL, μs * (1-α), μs * α, f, τ, n, β)) ./ μs ≈ C_elastic(λs, μs) ./ μs
        @test real(C_GT(λ, λ, ρs, vL, μs * (1-α), μs * α, f, τ, n, β)) ./ μs ≈ C_elastic(λs, μs) ./ μs
        @test real(C_C(λ, λ, ρs, vL, μs * (1-α), μs * α, f, τ, n, β)) ./ μs ≈ C_elastic(λs, μs) ./ μs

        # Strain-hardening regime models
        N = 2.2 # typical value for I1^N models
        @test real(C_MRSH(λ, λ, ρs, vL, μs * (0.9-α), μs * α, 0.1 * μs, N, f, τ, n, β)) ./ μs ≈ C_elastic(λs, μs) ./ μs
        @test real(C_GTSH(λ, λ, ρs, vL, μs * (0.9-α), μs * α, 0.1 * μs, N, f, τ, n, β)) ./ μs ≈ C_elastic(λs, μs) ./ μs
        @test real(C_CSH(λ, λ, ρs, vL, μs * (0.9-α), μs * α, 0.1 * μs, N, f, τ, n, β)) ./ μs ≈ C_elastic(λs, μs) ./ μs
        @test real(C_MRSH2(λ, λ, ρs, vL, μs * (0.9-α), μs * α, 0.1 * μs, N, f, τ, n, β)) ./ μs ≈ C_elastic(λs, μs) ./ μs
        @test real(C_GTSH2(λ, λ, ρs, vL, μs * (0.9-α), μs * α, 0.1 * μs, N, f, τ, n, β)) ./ μs ≈ C_elastic(λs, μs) ./ μs

        # Limiting-chain models
        J = 35 # typical value
        @test real(C_GMR(λ, λ, ρs, vL, μs * (1-α), μs * α, J, f, τ, n, β)) ./ μs ≈ C_elastic(λs, μs) ./ μs
        @test real(C_GG(λ + 1e-6, λ, ρs, vL, μs * (1-α), μs * α, J, f, τ, n, β)) ./ μs ≈ C_elastic(λs, μs) ./ μs
        @test real(C_DCMR2(λ + 1e-6, λ, ρs, vL, μs * (1-α), μs * α, J, f, τ, n, β)) ./ μs ≈ C_elastic(λs, μs) ./ μs
        @test real(C_DCGT(λ + 1e-6, λ, ρs, vL, μs * (1-α), μs * α, J, f, τ, n, β)) ./ μs ≈ C_elastic(λs, μs) ./ μs
        # For the last three models to work we need to use slightly different values of λ. Issue with the simplify in Mathematica?
    end

    # 2. Check that the discretized eigenvalue problem gives reference values.
    @testset "elastodynamics 1D" begin
        udof = 1:3
        λ1 = 1.3
        λ3 = 1 / sqrt(1.3)
        ρs = 1e3
        vL = 1e3
        μs = 23e3
        α = 0.3
        f = 50 
        τ = 330e-6
        n = 0.3
        β = 0.3 
        C = C_MR(λ1, λ3, ρs, vL, μs * (1 - α), μs * α, f, τ, n , β)
        C = C ./ μs

        N = 12
        x2, D = chebdif(N, 2)
        D2 = -2 * D[:, :, 1]
        D22 = 4 * D[:, :, 2]

        # 1D cartesian
        PEP1D = elastodynamics1D(C, D2, D22, udof)
        ref = load("PEP1D.jld2")
        for i in eachindex(PEP1D)
            @test real(ref["PEP"][i]) ≈ real(PEP1D[i])
            @test imag(ref["PEP"][i]) ≈ imag(PEP1D[i])
        end
    end

    @testset "elastodynamics 1D tube" begin
        udof = 1:3
        λ1 = 1.3
        λ3 = 1 / sqrt(1.3)
        ρs = 1e3
        vL = 1e3
        μs = 23e3
        α = 0.3
        f = 50 
        τ = 330e-6
        n = 0.3
        β = 0.3 
        C = C_MR(λ1, λ3, ρs, vL, μs * (1 - α), μs * α, f, τ, n , β)
        C = C ./ μs

        N = 12
        x2, D = chebdif(N, 2)
        R = 20e-3
        h = 1e-3
        x2 = (R .+ (h .* x2) ./ 2) ./ h
        D2 = 2 .* (h/h) * D[:,:,1]
        D22 = 4 .* (h/h)^2 * D[:,:,2]
        A = [0 0 0; 0 0 -1; 0 1 0] # Derivation of the basis vectors
        AA = (1im * n * I(length(udof)) .+ A) 

        PEP1Dtube = elastodynamics1D_tube(C, x2, D2, D22, AA, udof)
        ref = load("PEP1Dtube.jld2")
        for i in eachindex(PEP1Dtube)
            @test real(ref["PEP"][i]) ≈ real(PEP1Dtube[i])
            @test imag(ref["PEP"][i]) ≈ imag(PEP1Dtube[i])
        end
    end

    @testset "elastodynamics 2D" begin
        udof = 1:3
        λ1 = 1.3
        λ3 = 1 / sqrt(1.3)
        ρs = 1e3
        vL = 1e3
        μs = 23e3
        α = 0.3
        f = 50 
        τ = 330e-6
        n = 0.3
        β = 0.3 
        C = C_MR(λ1, λ3, ρs, vL, μs * (1 - α), μs * α, f, τ, n , β)
        C = C ./ μs

        h = 1e-3
        w = 1e-2
        N = 7 # number of nodes in direction 2 (odd, out-of-plane modes)
        P = 14 # number of nodes in direction 3
        _, Dt1 = chebdif(N,2)
        D2 = - 2 * (h/h) .* Dt1[:,:,1]
        D22 = 4 * (h/h)^2 .* Dt1[:,:,2]
        _, Dt2 = chebdif(P,2)
        D3 =  - 2 * h/w .* Dt2[:,:,1] # normalized with h
        D33 = 4 *(h/w)^2 .* Dt2[:,:,2]
        D23d = D3 ⊗ D2 
        D2d = I(P) ⊗ D2
        D22d = I(P) ⊗ D22
        D3d = D3 ⊗ I(N)
        D33d = D33 ⊗ I(N)

        PEP2D = elastodynamics2D(C, D2d, D3d, D22d, D33d, D23d, udof)
        ref = load("PEP2D.jld2")
        for i in eachindex(PEP2D)
            @test real(ref["PEP"][i]) ≈ real(PEP2D[i])
            @test imag(ref["PEP"][i]) ≈ imag(PEP2D[i])
        end
        
    end

    @testset "elastodynamics 2D polar" begin
        udof = 1:3
        λ1 = 1.3
        λ3 = 1 / sqrt(1.3) 
        ρs = 1070
        μs = 23e3
        vL = 1000
        α = 0.3
        f = 50 
        τ = 330e-6
        n = 0.3
        β = 0.3 
        h = 1e-3
        w = 1e-2
        R = 0.5*w # radius of curvature
        # Define elastic tensor
        C = C_MR(λ1, λ3, ρs, vL, μs * (1 - α), μs * α, f, τ, n , β)
        C = C ./ μs

        # Discretization: We differentiate in direction 2 and 3 using SCM.
        N = 5 # number of nodes in direction 2, must be even for in-plane modes and odd for out-of-plane modes
        P = 10 # number of nodes in direction 3
        x2, Dt1 = chebdif(N,2) # Nodes in [-1, 1]
        x2 = (R .+ (h .* x2) ./ 2) ./ h
        D2 = 2 .* (h/h) * Dt1[:,:,1]
        D22 = 4 .* (h/h)^2 * Dt1[:,:,2]
        _, Dt2 = chebdif(P,2)
        D3 = 2 * (R/w) .* Dt2[:,:,1] 
        D33 = 4 *(R/w)^2 .* Dt2[:,:,2]
        A = [0 0 0; 0 0 -1; 0 1 0] # Derivation of the basis vectors

        # Build the polynomial eigenvalue problem
        CPEP2D = Celastodynamics2D(C,x2,D2,D3,D22,D33,A,udof)
        ref = load("CPEP2D.jld2")
        for i in eachindex(CPEP2D)
            @test real(ref["PEP"][i]) ≈ real(CPEP2D[i])
            @test imag(ref["PEP"][i]) ≈ imag(CPEP2D[i])
        end
    end

end
