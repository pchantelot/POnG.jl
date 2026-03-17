# Guided wave dispersion in an elastic plate coupled to an infinite fluid domain using SCM 
# Load packages
# Easy identity matrix and kronecker tensor notation
using LinearAlgebra, Kronecker
# POnG
using POnG
# Plotting 
using CairoMakie
BLAS.set_num_threads(1)

## Problem definition: We look for Lamb waves propagating in direction 1 in a plate with thickness h  
# 
#            water: ρf, cf 
#  e2        ---------------------------------- top
#  |_ e1      solid: μs, λs, ρs, h
#            ---------------------------------- bottom
#            vacuum
# Choose degrees of freedom
udof = 1:3 # 1:2 -> Lamb modes only, 3:3 -> SH modes

# Solid properties 
ρs = 1070
μs = 23e3
vL = 1000
λs = vL^2*ρs - 2*μs # measured speed of sound in Ecoflex 1000 m/s
h = 1e-3
# Define the elastic tensor
C = C_elastic(λs, μs)
C = C ./ λs # normalization
# Fluid properties (Implementation relevant only if ct/cf -> 0)
ρf = 1000
cf = 1500 # speed of sound in water

# Discretization: We differentiate in direction 2 using SCM.
N = 12 # number of nodes
_, D = chebdif(N, 2)
D2 = - 2 * (h/h) * D[:,:,1]
D22 = 4 * (h/h)^2 * D[:,:,2]

# Build the polynomial eigenvalue problem
Lkk, Lk, L0, M = elastodynamics1D(C, D2, D22, udof)
M = convert(Array{ComplexF64},M) # We later add a complex to the mass matrix

# Top traction operator : 1 for BC in top, N for BC bottom
Bkt = C[2,udof,udof,1] ⊗ I(N)[1,:]'; Bkt = collect(Bkt)
B0t = C[2,udof,udof,2] ⊗ D2[1,:]'; B0t = collect(B0t)
# Bottom traction operator
Bkb = C[2,udof,udof,1] ⊗ I(N)[N,:]'; Bkb = collect(Bkb)
B0b = C[2,udof,udof,2] ⊗ D2[N,:]'; B0b = collect(B0b)

# Fluid addition
e2 = zeros(length(udof)); e2[2] = 1
Bρf = - 1im*ρf/ρs * I(length(udof)) .* e2 ⊗ I(N)[1,:]' 

## Impose BC
bc = [(0:length(udof)-1).*N .+ 1, (1:length(udof)).*N] #Location of boundary vectors in discretized matrix
# top BC 
Lkk[bc[1],:] .= Bkt 
Lk[bc[1],:] .= B0t
L0[bc[1],:] .= 0
M[bc[1],:] .= Bρf
# bottom BC 
Lkk[bc[2],:] .= Bkb
Lk[bc[2],:] .= B0b
L0[bc[2],:] .= 0
M[bc[2],:] .= 0

## Solve polynomial eigenvalue problem
# Impose ω and get eigenvalues and eigenvectors
ω = 2 * pi .* range(1e-2, 400, length = 70)
k = Array{ComplexF64,2}(undef, length(ω), 2 * length(udof) * N)
u = Array{ComplexF64,3}(undef, length(ω), length(udof) * N, 2 * length(udof) * N)

@time for i in eachindex(ω)
    local temp =  ω[i] * h /sqrt(λs/ρs)
    local problem = PEP([temp^2 .* M .+ L0, Lk, Lkk])
    local ktmp, utmp = polyeig(problem)
    k[i,:] = -1im * ktmp' / h
    u[i,:,:] = utmp
end

# Store result of a 1D calculation before (and for) postprocessing
result = result1D(udof,N,ω,k,u)
# Determine mode symmetry and principal displacement direction
idx = modedirection(result)
sym = modesymmetry(result)

## Plots
# Remove evanescent solutions
k[(1 .- abs.(imag.(k[:]))) .< 0] .= NaN
kplot = k[:]
idxplot = idx[:]
symplot = sym[:]
fplot = repeat(ω, 1, Int(2 * length(udof) * N))[:] / (2 * pi)

with_theme(theme_latexfonts()) do 
    set_theme!(Theme(palette = (color = [:red, :blue],),))

    fig = Figure(size = ( 1.2 * 360, 360))
        ax = Axis(fig[1,1])
        ax.limits = (0, 1000, 0, 300)
        ax.xlabel = L"k \,\, \mathrm{(1/m)}"
        ax.ylabel = L"f \,\, \mathrm{(Hz)}"
        ax.xlabelsize = 20 
        ax.ylabelsize = 20
        ax.title = "Lamb modes"
        scatter!(ax,real(kplot[symplot .!== 1 .&& idxplot .!== 3]),
            fplot[symplot .!== 1 .&& idxplot .!== 3],
            label = "Antisymmetric")
        scatter!(ax,real(kplot[symplot .== 1 .&& idxplot .!== 3]),
            fplot[symplot .== 1 .&& idxplot .!== 3],
            label = "Symmetric")
        axislegend(position= :rb)

    display(fig)
end
