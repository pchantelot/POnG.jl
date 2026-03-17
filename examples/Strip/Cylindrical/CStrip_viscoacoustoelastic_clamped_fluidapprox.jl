# Strip in cylindrical coordinates
# Load packages
# Easy identity matrix and kronecker tensor notation
using LinearAlgebra, Kronecker
# POnG
using POnG
# Plotting 
using CairoMakie
BLAS.set_num_threads(1)
using Base.Threads: @threads


## Problem definition: We look for Lamb waves propagating in direction 1 in a curved strip. 
# 
# Strip with height h, width w and curvature radius R in vaccuum
#
#                           top, fluid ρf, cf
#  e2 (= r)             ---------------------------------- 
#  |_ e3 (= θ)     left - solid: μs, λs, ρs              - right, zero displacement
#                       ---------------------------------- 
#                           bottom, zero normal traction

# Waves propagate in direction 1 (= x),  U(x,t) = u(r,θ)e^{i(kx-ωt)}
# normalization using ρs, μs and h
# Choose degrees of freedom
udof = 1:3 # displacement dof, coupled Lamb and SH waves

# Solid properties 
ρs = 1070
μs = 23e3
vL = 1000
λs = vL^2*ρs - 2*μs # measured speed of sound in Ecoflex 1000 m/s
h = 0.77e-3
w = 1e-2
R = 0.78 * w # radius of curvature
# Define parameters to compute the 4th order elastic tensor
λ1 = 1.
λ3 = 1.06
τ = 330e-6
n = 0.32
α = 0.29
β = 0.29
# update geometry (λ1λ2λ3 = 1)
hdef = h / (λ1*λ3)
wdef = w*λ3
# Fluid properties (Implementation relevant only if ct/cf -> 0)
ρf = 1000
cf = 1500 # speed of sound in water

# Discretization: We differentiate in direction 2 and 3 using SCM.
N = 5 # number of nodes in direction 2, must be even for in-plane modes and odd for out-of-plane modes
P = 10 # number of nodes in direction 3
x2, Dt1 = chebdif(N,2) # Nodes in [-1, 1]
x2 = (R .+ (hdef .* x2) ./ 2) ./ hdef
D2 = 2 .* (hdef/hdef) * Dt1[:,:,1]
D22 = 4 .* (hdef/hdef)^2 * Dt1[:,:,2]
_, Dt2 = chebdif(P,2)
D3 = 2 * (R/wdef) .* Dt2[:,:,1] 
D33 = 4 *(R/wdef)^2 .* Dt2[:,:,2]
A = [0 0 0; 0 0 -1; 0 1 0] # Derivation of the basis vectors

# boundary conditions
# N×P vector to N×P matrix
bc = reshape(1:N*P,N,P)
top = bc[1,:]
bottom = bc[N,:]
left = bc[:,1]
right = bc[:,P]
topall = (0:length(udof)-1)*N*P .+ top'; topall = topall'[:]
bottomall = (0:length(udof)-1)*N*P .+ bottom'; bottomall = bottomall'[:]
leftall = (0:length(udof)-1)*N*P .+ left'; leftall = leftall'[:]
rightall = (0:length(udof)-1)*N*P .+ right'; rightall = rightall'[:]

## Solve polynomial eigenvalue problem in k to get complex wavenumber
ω = 2 * pi .* range(1, 400, length = 70)
k = Array{ComplexF64,2}(undef, length(ω), 2 * length(udof) * N * P)
u = Array{ComplexF64,3}(undef, length(ω), length(udof) * N * P, 2 * length(udof) * N * P)
# create and solve polynomial eigenvalue problem
@time @threads for i in axes(ω,1)

    C = C_MR(λ1, λ3, ρs, vL, μs*(1-α), μs*α, ω[i] / (2*pi), τ, n, β)
    C = C ./ μs

    # Build the polynomial eigenvalue problem
    Lkk, Lk, L0, M = Celastodynamics2D(C,x2,D2,D3,D22,D33,A,udof)

    # boundary traction operators top
    Bkt = C[2,udof,udof,1] ⊗ I(N*P)[top,:]
    B0t = C[2,udof,udof,2] ⊗ (I(P) ⊗ D2)[top,:] .+ C[2,udof,udof,3] ⊗ (D3 ⊗ Diagonal(1 ./ x2))[top,:] .+ C[2,udof,udof,3] * A ⊗ (I(P) ⊗ Diagonal(1 ./ x2))[top,:]
    # boundary traction operators bottom
    Bkb = C[2,udof,udof,1] ⊗ I(N*P)[bottom,:]
    B0b = C[2,udof,udof,2] ⊗ (I(P) ⊗ D2)[bottom,:] .+ C[2,udof,udof,3] ⊗ (D3 ⊗ Diagonal(1 ./ x2))[bottom,:] .+ C[2,udof,udof,3]*A ⊗ (I(P) ⊗ Diagonal(1 ./ x2))[bottom,:]
    # zero displacement left
    B0l = I(length(udof)) ⊗ I(N*P)[left,:]
    # zero displacement right
    B0r = I(length(udof)) ⊗ I(N*P)[right,:]
    # Fluid coupling: 
    e2 = zeros(length(udof)); e2[2] = 1
    Bρft = 1im*e2.*I(3) ⊗ I(N*P)[top,:] .* ρf/ρs
    Bρfb =  -1im*e2.*I(3) ⊗ I(N*P)[bottom,:] .* ρf/ρs
    M = convert(Array{ComplexF64},M)

    # Impose BCs
    # top
    Lkk[topall,:] .= Bkt
    Lk[topall,:] .= B0t
    L0[topall,:] .= 0
    M[topall,:] .= 0
    # bottom
    Lkk[bottomall,:] .= Bkb
    Lk[bottomall,:] .= B0b
    L0[bottomall,:] .= 0
    M[bottomall,:] .= Bρfb
    # left
    Lkk[leftall,:] .= 0
    Lk[leftall,:] .= 0
    L0[leftall,:] .= B0l
    M[leftall,:] .= 0
    # right
    Lkk[rightall,:] .= 0
    Lk[rightall,:] .= 0
    L0[rightall,:] .= B0r
    M[rightall,:] .= 0

    local temp = ω[i] * hdef / sqrt(μs/ρs)
    local problem = PEP([temp^2 .* M .+ L0, Lk, Lkk])
    local ktmp, utmp = polyeig(problem)
    k[i,:] = -1im * ktmp' / hdef
    u[i,:,:] = utmp

end

# Store result of a 2D calculation before (and for) postprocessing
result = result2D(udof,N,P,ω,k,u)
#save(joinpath(@__DIR__,"Alex_fig410_A0.jld2"),"data",result)
idx = modedirection(result)

# remove negative real(k) and purely imaginary solutions
f = repeat(ω,1,Int(2*(length(udof)*N*P))) /(2*pi)
#k[(1 .- abs.(imag.(k[:]))) .< 0] .= NaN
k[(0.49*pi .< angle.(k)) .| (angle.(k) .< -0.49*pi)] .= NaN 
k[imag(k) .> 0 ] .= NaN
k[real(k) .> 1400] .= NaN
k[idx .!== 2] .= NaN
f[isnan.(k)] .= NaN
idx[isnan.(k)] .= 4

## Plots
kplot = k[:]
fplot = f[:]
idxplot = idx[:]
filter!(!isnan,kplot)
filter!(!isnan,fplot)
filter!(x -> x .!== 4,idxplot)
shading  = 1 .- abs.(imag.(kplot)) / (0.04*maximum(abs.(imag.(kplot))))
shading[shading .< 0] .= NaN
fplot[isnan.(shading)] .= NaN
kplot[isnan.(shading)] .= NaN
filter!(!isnan,fplot)
filter!(!isnan,kplot)
filter!(!isnan,shading)

with_theme(theme_latexfonts()) do 

    fig = Figure(size = (1.2 * 360, 360))

        ax = Axis(fig[1,1])
        ax.limits = (0, 1000, 0, 350)
        ax.xlabel = L"k \,\, \mathrm{(1/m)}"
        ax.ylabel = L"f \,\, \mathrm{(Hz)}"
        ax.xlabelsize = 20 
        ax.ylabelsize = 20
        ax.title = "Out of plane modes"
        scatter!(ax,real(kplot),fplot, color = tuple.(:blue,shading))

    display(fig)

end
