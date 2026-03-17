# Guided wave dispersion in an elastic strip using SCM 
# Load packages
# Easy identity matrix and kronecker tensor notation
using LinearAlgebra, Kronecker
# Plotting 
using CairoMakie
# POnG
using POnG
BLAS.set_num_threads(1)
using Base.Threads: @threads


## Problem definition: We look for Lamb waves propagating in direction 1 in a strip with 
#height h and width w in vaccuum.
#
#                               top
#  e2             ---------------------------------- 
#  |_ e3     left - solid: μs, λs, ρs              - right
#                 ---------------------------------- 
#                               bottom
# Waves propagate in direction 1, U(x,t) = u(y,z)e^{i(kx-ωt)}
# normalization using ρs, μs and h

# displacement dof
udof = 1:3 
# Solid properties (Ecoflex OO-30)
ρs = 1070
μs = 23e3
h = 1e-3
w = 1e-2
vL = 1000
λs = vL^2*ρs - 2*μs # measured speed of sound in Ecoflex 1000 m/s 
# Define elastic tensor
C = C_elastic(λs, μs)
C = C ./ μs # normalization

# Discretization: We differentiate in direction 2 and 3 using SCM.
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

# Build the polynomial eigenvalue problem
Lkk, Lk, L0, M = elastodynamics2D(C, D2d, D3d, D22d, D33d, D23d, udof)

# boundary conditions
# N×P vector to N×P matrix
bc = reshape(1:N*P,N,P)
top = bc[1,:]
bottom = bc[N,:]
left = bc[:,1]
right = bc[:,P]
# boundary traction operators top
Bkt = C[2,udof,udof,1] ⊗ I(N*P)[top,:]
B0t = C[2,udof,udof,2] ⊗ D2d[top,:] .+ C[2,udof,udof,3] ⊗ D3d[top,:]
# boundary traction operators bottom
Bkb = C[2,udof,udof,1] ⊗ I(N*P)[bottom,:]
B0b = C[2,udof,udof,2] ⊗ D2d[bottom,:] .+ C[2,udof,udof,3] ⊗ D3d[bottom,:]
# boundary traction operators left
Bkl = C[3,udof,udof,1] ⊗ I(N*P)[left,:]
B0l = C[3,udof,udof,2] ⊗ D2d[left,:] .+ C[3,udof,udof,3] ⊗ D3d[left,:]
# boundary traction operators right
Bkr = C[3,udof,udof,1] ⊗ I(N*P)[right,:]
B0r = C[3,udof,udof,2] ⊗ D2d[right,:] .+ C[3,udof,udof,3] ⊗ D3d[right,:]

# Impose boundary conditions
topall = (0:length(udof)-1)*N*P .+ top'; topall = topall'[:]
bottomall = (0:length(udof)-1)*N*P .+ bottom'; bottomall = bottomall'[:]
leftall = (0:length(udof)-1)*N*P .+ left'; leftall = leftall'[:]
rightall = (0:length(udof)-1)*N*P .+ right'; rightall = rightall'[:]

# top
Lkk[topall,:] .= 0
Lk[topall,:] .= Bkt
L0[topall,:] .= B0t
M[topall,:] .= 0
# bottom
Lkk[bottomall,:] .= 0
Lk[bottomall,:] .= Bkb
L0[bottomall,:] .= B0b
M[bottomall,:] .= 0
# left
Lkk[leftall,:] .= 0
Lk[leftall,:] .= Bkl
L0[leftall,:] .= B0l 
M[leftall,:] .= 0
# right
Lkk[rightall,:] .= 0
Lk[rightall,:] .= Bkr
L0[rightall,:] .= B0r
M[rightall,:] .= 0

## Solve polynomial eigenvalue problem
# Impose ω and get eigenvalues and eigenvectors
ω = 2*pi .* range(1e-3, 400, length = 100) 
k = Array{ComplexF64,2}(undef,length(ω),2*length(udof)*N*P)
u = Array{ComplexF64,3}(undef,length(ω),length(udof)*N*P,2*length(udof)*N*P)

@time @threads for i in eachindex(ω)
    local temp = ω[i] * h / sqrt(μs/ρs)
    local problem = PEP([temp^2 .* M .+ L0, Lk, Lkk])
    k[i,:] , u[i,:,:] = polyeig(problem)
end

k = -1im * k / h
# Store result of a 2D calculation before (and for) postprocessing
result = result2D(udof,N,P,ω,k,u)
idx = modedirection(result)

## Plots
# remove negative real(k) and purely imaginary solutions
f = repeat(ω,1,Int(2*(length(udof)*N*P))) /(2*pi)
k[(1 .- abs.(imag.(k[:]))) .< 0] .= NaN
f[isnan.(k)] .= NaN
kplot = k[:]
fplot = f[:]
idxplot = idx[:]

with_theme(theme_latexfonts()) do 

    fig = Figure(size = (1.2 * 360, 360))

        ax = Axis(fig[1,1])
        ax.limits = (0, 1100, 0, 400)
        ax.xlabel = L"k \,\, \mathrm{(1/m)}"
        ax.ylabel = L"f \,\, \mathrm{(Hz)}"
        ax.xlabelsize = 20 
        ax.ylabelsize = 20
        ax.title = "Out of plane modes"
        scatter!(ax,real(kplot[idxplot .== 2]),fplot[idxplot .== 2],
        markersize = 10, color = :red)

    display(fig)
end