# Guided wave dispersion in an elastic plate coupled to an infinite fluid domain using SCM 
# Load packages
# Easy identity matrix and kronecker tensor notation
using LinearAlgebra, Kronecker
# POnG
using POnG
# Plotting 
using CairoMakie
# padding for fluid 
using PaddedViews
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
# Fluid properties
ρf = 1000
cf = 1500 # speed of sound in water

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

# Fluid coupling: pad matrices to add the equation for the extra degree of freedom uf
Lkk = collect(PaddedView(0,Lkk,size(Lkk).+1))
Lk = collect(PaddedView(0,Lk,size(Lk).+1)); Lk = convert(Array{ComplexF64}, Lk)
L0 = collect(PaddedView(0,L0,size(L0).+1))
M = collect(PaddedView(0,M,size(M).+1))
Bkt = collect(PaddedView(0,Bkt, (size(Bkt)[1], size(Bkt)[2]+1)))
B0t = collect(PaddedView(0,B0t, (size(B0t)[1], size(B0t)[2]+1)))
Bkb = collect(PaddedView(0,Bkb, (size(Bkb)[1], size(Bkb)[2]+1)))
B0b = collect(PaddedView(0,B0b, (size(B0b)[1], size(B0b)[2]+1)))
# add the extra equation for uf (extra row) and the normal stress BC (extra column)
e2 = zeros(length(udof)); e2[2] = 1
e2dt = e2' ⊗ I(N)[1,:]' # discretized e2
Bωf = zeros(length(udof)*N+1,length(udof)*N+1) # initialize  Bωf
Bωf[end,:] = ρf*cf/(ρs*sqrt(λs/ρs)) .* hcat(e2dt,0) 
Bωf[:,end] = ρf*cf/(ρs*sqrt(λs/ρs)) .* vcat(e2dt',0) 

## to impose BC later
bc = [(0:length(udof)-1).*N .+ 1, (1:length(udof)).*N] #Location of boundary vectors in discretized matrix

## Solve polynomial eigenvalue problem
# Impose ω and get eigenvalues and eigenvectors
ω = 2 * pi .* range(1e-2, 400, length = 70)
k = Array{ComplexF64,2}(undef, length(ω), 4 * (length(udof) * N + 1))
ky = Array{ComplexF64,2}(undef, length(ω), 4 * (length(udof) * N + 1))
u = Array{ComplexF64,3}(undef, length(ω), length(udof) * N + 1, 4 * (length(udof) * N + 1))

@time for i in eachindex(ω)
    local temp =  ω[i] * h / sqrt(λs/ρs)
    # change of variables 
    local kf = temp / cf * sqrt(λs/ρs)
    local Gγγ = similar(Lkk)
    local Gγbc = similar(Lk)
    local G0bc = similar(L0)
    local Gmγbc = similar(Lk)

    Gγγ = -(kf/2)^2 * Lkk
    G0bc = L0 .+ 2*Gγγ .+ temp^2 * M
    Gγbc = 1im*kf/2 * Lk
    Gmγbc = 1im*kf/2 * Lk

    # top BC 
    Gγγ[bc[1],:] .= 0 
    Gγbc[bc[1],:] .= 1im*kf/2*Bkt 
    G0bc[bc[1],:] .= B0t .+ temp .* Bωf[bc[1],:] 
    Gmγbc[bc[1],:] .= 1im*kf/2*(Bkt) 
    # bottom BC 
    Gγγ[bc[2],:] .= 0 
    Gγbc[bc[2],:] .= 1im*kf/2*(Bkb)
    G0bc[bc[2],:] .= B0b .+ temp .* Bωf[bc[2],:] # Fluid only on top so nothing in Bωf
    Gmγbc[bc[2],:] .= 1im*kf/2*(Bkb)
    # equation for uf
    G0bc[end,:] .= temp * Bωf[end,:] 
    Gγbc[end,end] = -kf/2 * ρf*cf^2/λs
    Gmγbc[end,end] = kf/2 * ρf*cf^2/λs

    local problem = PEP([Gγγ, Gmγbc, G0bc, Gγbc, Gγγ])
    local ktmp, utmp = polyeig(problem)
    k[i,:] = kf / 2 * (ktmp .+ 1 ./ ktmp)' / h
    ky[i,:] = 1im * kf / 2 * (ktmp .- 1 ./ ktmp)' / h
    u[i,:,:] = utmp
end

# Store result of a 1D calculation before (and for) postprocessing
result = result1D(udof,N,ω,k,u)
# Determine mode symmetry and principal displacement direction
idx = modedirection(result)
sym = modesymmetry(result)

## Plots
k[(1 .- abs.(imag.(k[:]))) .< 0] .= NaN
k[real(ky) .< -1e-6] .= NaN
k[imag(ky) .<= 0] .= NaN
kplot = k[:]
idxplot = idx[:]
symplot = sym[:]
fplot = repeat(ω, 1, Int(4 * (length(udof) * N + 1)))[:] / (2 * pi)

with_theme(theme_latexfonts()) do 
    set_theme!(Theme(palette = (color = [:red, :blue],),))

    fig = Figure(size = (1.2 * 360, 360))
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
