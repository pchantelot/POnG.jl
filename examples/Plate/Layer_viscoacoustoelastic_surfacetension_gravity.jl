# Guided wave dispersion in a stretched viscoelastic layer using SCM 
# Load packages
# Easy identity matrix and kronecker tensor notation
using LinearAlgebra, Kronecker
# POnG
using POnG
# Plotting 
using CairoMakie
BLAS.set_num_threads(1)

## Problem definition: We look for guided waves propagating in direction 1 in a 2D layer with thickness h
# 
#  e2        ---------------------------------- top: free surface + surface tension γ
#  |_ e1      solid: μs, λs, ρs, h
#            ---------------------------------- bottom: no slip
# Choose degrees of freedom
udof = 1:3 # 1:2 -> Lamb modes only, 3:3 -> SH modes 

# Solid properties 
μs = 380
ρs = 1000
νs = 0.499984
λs = 2*μs*νs/(1-2*νs)
vL = sqrt((λs+2*μs)/ρs)
h = 9.8e-3
# Define parameters to compute the 4th order elastic tensor 
λ1 = 1. 
λ3 = 1. 
hdef = h /(λ1*λ3)
τ = 10e-6
n = 0.32
α = 0. # Neo-Hookean
β = 0.29
# surface tension
γ = 0.07
Γ = γ/(μs*hdef)
# gravity
g =  9.81
G = ρs*g*hdef/μs 

# Discretization: We differentiate in direction 2 using SCM.
N = 12 # number of nodes
_, D = chebdif(N, 2)
D2 = - 2 * (hdef/hdef) * D[:,:,1]
D22 = 4 * (hdef/hdef)^2 * D[:,:,2]
#Location of boundary vectors in discretized matrix
bc = [(0:length(udof)-1).*N .+ 1, (1:length(udof)).*N] 

## Solve polynomial eigenvalue problem
# Impose ω and get eigenvalues and eigenvectors
ω = 2*pi .* range(1e-2,150, length = 100)
k = Array{ComplexF64,2}(undef, length(ω), 2 * length(udof) * N)
u = Array{ComplexF64,3}(undef, length(ω), length(udof) * N, 2 * length(udof) * N)
# create and solve polynomial eigenvalue problem
@time for i in axes(ω,1)

    C = C_MR(λ1, λ3, ρs, vL, μs * (1-α), μs * α, ω[i] / (2*pi), τ, n, β)
    C = C ./ μs

    # Build the polynomial eigenvalue problem
    Lkk, Lk, L0, M = elastodynamics1D(C, D2, D22, udof)

    # Top traction operator : 1 for BC in top, N for BC bottom
    Bkt = C[2,udof,udof,1] ⊗ I(N)[1,:]'; Bkt = collect(Bkt)
    B0t = C[2,udof,udof,2] ⊗ D2[1,:]'; B0t = collect(B0t)
    # surface tension
    e2 = zeros(length(udof)); e2[2] = 1
    Bkkt = Γ .* I(length(udof)) .* e2 ⊗ I(N)[1,:]'
    # add gravity as a free surface contribution
    B0t .= B0t .- G .* I(length(udof)) .* e2 ⊗ I(N)[1,:]'
    # Bottom zero displacement
    B0b = I(length(udof)) ⊗ I(N)[N,:]'; B0b = collect(B0b)

    ## Impose BC
    # top BC 
    Lkk[bc[1],:] .= Bkkt
    Lk[bc[1],:] .= Bkt
    L0[bc[1],:] .= B0t
    M[bc[1],:] .= 0
    # bottom BC 
    Lkk[bc[2],:] .= 0
    Lk[bc[2],:] .= 0
    L0[bc[2],:] .= B0b
    M[bc[2],:] .= 0

    ## Solve polynomial eigenvalue problem in k to get complex wavenumber
    # create and solve polynomial eigenvalue problem
    local temp = ω[i] * hdef /sqrt(μs/ρs)
    local problem = PEP([temp^2 .* M .+ L0, Lk, Lkk])
    local ktmp, utmp = polyeig(problem)
    k[i,:] = -1im * ktmp' / hdef
    u[i,:,:] = utmp

end

# Store result of a 1D calculation before (and for) postprocessing
result = result1D(udof,N,ω,k,u)
# Determine mode symmetry and principal displacement direction
idx = modedirection(result)
sym = modesymmetry(result)
#pol = modepolarisation(result)

## Plots
# Remove evanescent solutions
k[(0.49*pi .< angle.(k)) .| (angle.(k) .< -0.49*pi)] .= NaN 
# remove large spurious k to improve shading
k[real(k) .> 1500] .= NaN
shading  = 1 .- abs.(imag.(k[:])) / (0.05 * maximum(abs.(imag.(filter(!isnan,k[:])))))

kplot = k[:]
fplot = repeat(ω, 1, Int(2 * length(udof) * N))[:] / (2 * pi)
idxplot = idx[:]
splot = shading[:]
#polplot = pol[:]


with_theme(theme_latexfonts()) do 

    fig = Figure(size = (1.2 * 360, 360))
        ax = Axis(fig[1,1])
        ax.limits = (0, 1500, 0, 140)
        ax.xlabel = L"k \,\, \mathrm{(1/m)}"
        ax.ylabel = L"f \,\, \mathrm{(Hz)}"
        ax.xlabelsize = 20 
        ax.ylabelsize = 20
        ax.title = "Lamb modes"
        s = scatter!(ax, real(kplot[idxplot .!== 3]), fplot[idxplot .!== 3],
            color = tuple.(:red, splot[idxplot .!== 3]))
        #Colorbar(fig[1,2], s, label = "Polarisation: in-plane → out-of-plane")

    display(fig)

end