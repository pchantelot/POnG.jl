# Guided wave dispersion in a stretched viscoelastic plate using SCM 
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
#  e2        ---------------------------------- top
#  |_ e1      solid: μs, λs, ρs, h
#            ---------------------------------- bottom
#
# Choose degrees of freedom
udof = 1:3 # 1:2 -> Lamb waves only, 3 -> SH waves 

# Solid properties (Ecoflex OO-30)
ρs = 1070
μs = 23e3
vL = 1000
λs = vL^2*ρs - 2*μs # measured speed of sound in Ecoflex 1000 m/s
h = 1e-3
# Define parameters to compute the 4th order elastic tensor
λ1 = 1.0
λ3 = 1 / sqrt(λ1)
τ = 330e-6
n = 0.32
α = 0.29
β = 0.29
# update geometry (λ1λ2λ3 = 1)
hdef = h / (λ1*λ3)

# Discretization: We differentiate in direction 2 using SCM.
N = 12 # number of nodes
_, D = chebdif(N, 2)
D2 = - 2 * (hdef/hdef) * D[:,:,1]
D22 = 4 * (hdef/hdef)^2 * D[:,:,2]
# Define location of boundary elements in the discrete matrix
bc = [(0:length(udof)-1) .* N .+ 1, (1:length(udof)) .* N] 

## Solve polynomial eigenvalue problem
# Impose ω and get eigenvalues and eigenvectors
ω = 2 * pi * sqrt(μs/ρs) / h * range(1e-6, 3, length = 70) 
k = Array{ComplexF64,2}(undef, length(ω), 2 * length(udof) * N)
u = Array{ComplexF64,3}(undef, length(ω), length(udof) * N, 2 * length(udof) * N)

@time for i in axes(ω, 1)
    
    # Frequency dependent viscoelastic tensor
    C = C_MR(λ1, λ3, ρs, vL, μs * (1-α), μs * α, ω[i] / (2*pi), τ, n, β)
    C = C ./ μs

    # Build the polynomial eigenvalue problem
    Lkk, Lk, L0, M = elastodynamics1D(C, D2, D22, udof)

    # Top traction operator : 1 for BC in top, N for BC bottom
    Bkt = C[2,udof,udof,1] ⊗ I(N)[1,:]'; Bkt = collect(Bkt)
    B0t = C[2,udof,udof,2] ⊗ D2[1,:]'; B0t = collect(B0t)
    # Bottom traction operator
    Bkb = C[2,udof,udof,1] ⊗ I(N)[N,:]'; Bkb = collect(Bkb)
    B0b = C[2,udof,udof,2] ⊗ D2[N,:]'; B0b = collect(B0b)

    # Impose BC: location of boundary vectors in discretized matrix
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

    ## Solve polynomial eigenvalue problem in k to get complex wavenumber
    # create and solve polynomial eigenvalue problem
    temp =  ω[i] ./ sqrt(μs/ρs) .* hdef
    problem = PEP([temp^2 .* M .+ L0, Lk, Lkk])
    ktmp, utmp = polyeig(problem)
    k[i,:] = - 1im * ktmp' / hdef
    u[i,:,:] = utmp
 
end

# Store result of a 1D calculation before (and for) postprocessing
result = result1D(udof,N,ω,k,u)
# Determine mode symmetry and principal displacement direction
idx = modedirection(result)
sym = modesymmetry(result)

# remove negative real(k) and purely imaginary solutions
k[(0.49*pi .< angle.(k)) .| (angle.(k) .< -0.49*pi)] .= NaN 
# remove large spurious k to improve shading
k[real(k) .> 20/h] .= NaN
shading  = 1 .- abs.(imag.(k[:])) / (0.2 * maximum(abs.(imag.(filter(!isnan,k[:])))))

kplot = k[:]
idxplot = idx[:]
symplot = sym[:]
fplot = repeat(ω, 1, Int(2 * length(udof) * N))[:] / (2 * pi)
splot = shading[:]

with_theme(theme_latexfonts()) do 
    set_theme!(Theme(palette = (color = [:red, :blue],),))

    fig = Figure(size = (2*1.2*360, 360))

        ax = Axis(fig[1,1])
        ax.limits = (0, 10, 0, 3)
        ax.xlabel = L"kh"
        ax.ylabel = L"fh/c_t"
        ax.yticks = [0, 0.5, 1, 1.5, 2, 2.5, 3]
        ax.xlabelsize = 20 
        ax.ylabelsize = 20
        ax.title = "Lamb modes"
        scatter!(ax,real(kplot[symplot .!== 1 .&& idxplot .!== 3]) * h,
            fplot[symplot .!== 1 .&& idxplot .!== 3] * h / sqrt(μs / ρs),
            color = tuple.(:red, splot[symplot .!== 1 .&& idxplot .!== 3]),
            label = "Antisymmetric")
        scatter!(ax,real(kplot[symplot .== 1 .&& idxplot .!== 3]) * h,
            fplot[symplot .== 1 .&& idxplot .!== 3] * h / sqrt(μs / ρs),
            color = tuple.(:blue, splot[symplot .== 1 .&& idxplot .!== 3]),
            label = "Symmetric")
        axislegend(position= :rb)

        ax2 = Axis(fig[1,2])
        ax2.limits = (0, 10, 0, 3)
        ax2.xlabel = L"kh"
        ax2.ylabel = L"fh/c_t"
        ax2.yticks = [0, 0.5, 1, 1.5, 2, 2.5, 3]
        ax2.xlabelsize = 20 
        ax2.ylabelsize = 20
        ax2.title = "SH modes"
        scatter!(ax2,real(kplot[idxplot .== 3]) * h,
            fplot[idxplot .== 3] * h / sqrt(μs / ρs),
            color = tuple.(:red, splot[idxplot .== 3]))

    display(fig)
end
