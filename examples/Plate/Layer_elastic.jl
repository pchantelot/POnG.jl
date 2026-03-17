# Guided wave dispersion in an elastic layer using SCM 
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
#  e2        ---------------------------------- top: free surface
#  |_ e1      solid: μs, λs, ρs, h
#            ---------------------------------- bottom: no slip

# Choose degrees of freedom
udof = 1:3 # 1:2 -> Lamb modes only, 3:3 -> SH modes 

# Solid properties 
μs = 380
ρs = 1000
νs = 0.499984
λs = 2*μs*νs/(1-2*νs)
h = 9.8e-3
# Define the elastic tensor
C = C_elastic(λs, μs)
C = C ./ μs # normalization

# Discretization: We solve elastodynamics by discretizing the elastodynamic equation in direction 2 using SCM.
N = 12 # number of nodes
_, D = chebdif(N, 2)
D2 = - 2 * (h/h) * D[:,:,1]
D22 = 4 * (h/h)^2 * D[:,:,2]

# Build the polynomial eigenvalue problem
Lkk, Lk, L0, M = elastodynamics1D(C, D2, D22, udof)

# Top traction operator : 1 for BC in top, N for BC bottom
Bkt = C[2,udof,udof,1] ⊗ I(N)[1,:]'; Bkt = collect(Bkt)
B0t = C[2,udof,udof,2] ⊗ D2[1,:]'; B0t = collect(B0t)
# Bottom no slip boundary conditions
B0b = I(length(udof)) ⊗ I(N)[N,:]'; B0b = collect(B0b)

## Impose BC
bc = [(0:length(udof)-1).*N .+ 1, (1:length(udof)).*N] #Location of boundary vectors in discretized matrix
# top BC 
Lkk[bc[1],:] .= 0 
Lk[bc[1],:] = Bkt
L0[bc[1],:] = B0t 
M[bc[1],:] .= 0
# bottom BC 
Lkk[bc[2],:] .= 0
Lk[bc[2],:] .= 0
L0[bc[2],:] = B0b 
M[bc[2],:] .= 0

## Solve polynomial eigenvalue problem
# Impose ω and get eigenvalues and eigenvectors
ω = 2*pi .* range(1e-2,150, length = 100)
k = Array{ComplexF64,2}(undef, length(ω), 2 * length(udof) * N)
u = Array{ComplexF64,3}(undef, length(ω), length(udof) * N, 2 * length(udof) * N)

@time for i in eachindex(ω)
    local temp =  ω[i] * h / sqrt(μs/ρs)
    local problem = PEP([temp^2 .* M .+ L0, Lk, Lkk])
    local ktmp, utmp = polyeig(problem)
    k[i,:] = -1im * ktmp' / h
    u[i,:,:] = utmp
end

# Store result of a 1D calculation before (and for) postprocessing
result = result1D(udof,N,ω,k,u)
idx = modedirection(result)

## Plots
# Remove evanescent solutions
k[(1 .- abs.(imag.(k[:]))) .< 0] .= NaN
kplot = k[:]
fplot = repeat(ω, 1, Int(2 * length(udof) * N))[:] / (2 * pi)
idxplot = idx[:]

with_theme(theme_latexfonts()) do 

    fig = Figure(size = ( 1.2 * 360, 360))
        ax = Axis(fig[1,1])
        ax.limits = (0, 1500, 0, 140)
        ax.xlabel = L"k \,\, \mathrm{(1/m)}"
        ax.ylabel = L"f \,\, \mathrm{(Hz)}"
        ax.xlabelsize = 20 
        ax.ylabelsize = 20
        scatter!(ax, real(kplot[idxplot .!== 3]), fplot[idxplot .!== 3])

    display(fig)
end

