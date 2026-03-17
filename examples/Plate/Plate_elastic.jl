# Guided wave dispersion in an elastic plate using SCM 
# Load packages
using LinearAlgebra, Kronecker
# POnG
using POnG
# Plotting 
using CairoMakie
BLAS.set_num_threads(1)

## Problem definition: We look for Lamb waves propagating in direction 1 in a plate with thickness h 
# 
#  e2        ---------------------------------- top
#  |_ e1      solid: μs, λs, ρs, h
#            ---------------------------------- bottom
#
# Choose degrees of freedom
udof = 1:3 # 1:2 -> Lamb modes only, 3:3 -> SH modes

# Solid properties (Ecoflex OO-30)
ρs = 1070
μs = 23e3
vL = 1000
λs = vL^2*ρs - 2*μs # measured speed of sound in Ecoflex 1000 m/s
h = 1e-3
# Define the elastic tensor
C = C_elastic(λs, μs)
C = C ./ μs # normalization

# Discretization: We differentiate in direction 2 using SCM.
N = 12 # number of nodes
_, D = chebdif(N, 2)
D2 = - 2 * (h/h) * D[:,:,1]
D22 = 4 * (h/h)^2 * D[:,:,2]

# Build the polynomial eigenvalue problem
Lkk, Lk, L0, M = elastodynamics1D(C, D2, D22, udof)

# Top traction operator : 1 for BC in top, N for BC bottom
Bkt = C[2,udof,udof,1] ⊗ I(N)[1,:]'; Bkt = collect(Bkt)
B0t = C[2,udof,udof,2] ⊗ D2[1,:]'; B0t = collect(B0t)
# Bottom traction operator
Bkb = C[2,udof,udof,1] ⊗ I(N)[N,:]'; Bkb = collect(Bkb)
B0b = C[2,udof,udof,2] ⊗ D2[N,:]'; B0b = collect(B0b)

# Impose BC: location of boundary vectors in discretized matrix
bc = [(0:length(udof)-1) .* N .+ 1, (1:length(udof)) .* N] 
# top BC 
Lkk[bc[1],:] .= 0 
Lk[bc[1],:] = Bkt
L0[bc[1],:] = B0t 
M[bc[1],:] .= 0
# bottom BC 
Lkk[bc[2],:] .= 0
Lk[bc[2],:] = Bkb
L0[bc[2],:] = B0b 
M[bc[2],:] .= 0

## Solve polynomial eigenvalue problem
# Impose ω and get eigenvalues and eigenvectors
ω = 2 * pi * sqrt(μs/ρs) / h * range(1e-6, 3, length = 100) 
k = Array{ComplexF64,2}(undef, length(ω), 2 * length(udof) * N)
u = Array{ComplexF64,3}(undef, length(ω), length(udof) * N, 2 * length(udof) * N)

@time for i in eachindex(ω)
    local temp = ω[i] * h / sqrt(μs/ρs)
    local problem = PEP([temp^2 .* M .+ L0, Lk, Lkk])
    k[i,:], u[i,:,:] = polyeig(problem)
end

k = -1im * k / h
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

##
with_theme(theme_latexfonts()) do 
    set_theme!(Theme(palette = (color = [:red, :blue],),))

    fig = Figure(size = (2 * 1.2 * 360, 360))

        ax = Axis(fig[1,1])
        ax.limits = (0, 20, 0, 3)
        ax.xlabel = L"kh"
        ax.ylabel = L"fh/c_t"
        ax.yticks = [0, 0.5, 1, 1.5, 2, 2.5, 3]
        ax.xlabelsize = 20 
        ax.ylabelsize = 20
        ax.title = "Lamb modes"
        scatter!(ax,real(kplot[symplot .!== 1 .&& idxplot .!== 3]) * h,
            fplot[symplot .!== 1 .&& idxplot .!== 3] * h / sqrt(μs / ρs),
            label = "Antisymmetric")
        scatter!(ax,real(kplot[symplot .== 1 .&& idxplot .!== 3]) * h,
            fplot[symplot .== 1 .&& idxplot .!== 3] * h / sqrt(μs / ρs),
            label = "Symmetric")
        axislegend(position= :rb)
        colsize!(fig.layout, 1, Aspect(1, 1.2))

        ax2 = Axis(fig[1,2])
        ax2.limits = (0, 20, 0, 3)
        ax2.xlabel = L"kh"
        ax2.ylabel = L"fh/c_t"
        ax2.yticks = [0, 0.5, 1, 1.5, 2, 2.5, 3]
        ax2.xlabelsize = 20 
        ax2.ylabelsize = 20
        ax2.title = "SH modes"
        scatter!(ax2,real(kplot[idxplot .== 3]) * h,
            fplot[idxplot .== 3] * h / sqrt(μs / ρs))
        colsize!(fig.layout, 2, Aspect(1, 1.2))

        resize_to_layout!(fig)
    display(fig)
end
