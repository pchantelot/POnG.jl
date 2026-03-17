# Guided wave dispersion in a stretched viscoelastic pipe in cylindrical coordinates
# Load packages
# Easy identity matrix and kronecker tensor notation
using LinearAlgebra, Kronecker
# Plotting 
using CairoMakie
# POnG
using POnG
BLAS.set_num_threads(1)

## Problem definition: We look for Lamb waves propagating in direction 1 in a cylinder with mean radius R with wall thickness h.
#
#                           top, zero normal traction
#  e2 (= r)             ---------------------------------- 
#  |_ e3 (= θ)          - solid: μs, λs, ρs              - 
#                       ---------------------------------- 
#                           bottom, zero normal traction

# Waves propagate in direction 1 (= x),  U(x,t) = u(r)e^{i(kx+nθ-ωt)}
# normalization using ρs, μs and h
udof = 1:3 # displacement dof, coupled Lamb and SH waves

# Solid properties 
ρs = 1070
μs = 23e3
vL = 1000
λs = vL^2*ρs - 2*μs # measured speed of sound in Ecoflex 1000 m/s
R = 10e-3 #radius of curvature
h = 1e-3
n = 0
# Define parameters to compute the 4th order elastic tensor
λ1 = 1.0
λ3 = 1.0
τ = 330e-6
nn = 0.32
α = 0.29
β = 0.29
# update geometry (λ1λ2λ3 = 1)
hdef = h / (λ1 * λ3)
Rdef = R * λ3

# Discretization: We differentiate in direction 2 using SCM.
N = 12 # number of nodes in direction 2
x2, D = chebdif(N, 2) # Nodes in [-1, 1]
x2 = (Rdef .+ (hdef .* x2) ./ 2) ./ hdef
D2 = 2 .* (hdef/hdef) * D[:,:,1]
D22 = 4 .* (hdef/hdef)^2 * D[:,:,2]
A = [0 0 0; 0 0 -1; 0 1 0] # Derivation of the basis vectors
AA = (1im * n * I(length(udof)) .+ A) 
# Impose BC: location of boundary vectors in discretized matrix
bc =  [(0:length(udof)-1).*N .+ 1, (1:length(udof)).*N]

## Solve polynomial eigenvalue problem
# Impose ω and get eigenvalues and eigenvectors
ω = 2 * pi * range(1e-3, 400, length = 70) 
k = Array{ComplexF64,2}(undef, length(ω), 2 * length(udof) * N)
u = Array{ComplexF64,3}(undef, length(ω), length(udof) * N, 2 * length(udof) * N)

@time for i in axes(ω,1)

    C = C_MR(λ1, λ3, ρs, vL, μs * (1-α), μs * α, ω[i], τ, nn, β)
    C = C ./ λs

    # Build the polynomial eigenvalue problem
    Lkk, Lk, L0, M = elastodynamics1D_tube(C,x2,D2,D22,AA,udof)

    # boundary traction operators top
    Bkt = C[2,udof,udof,1] ⊗ I(N)[1,:]'
    B0t = C[2,udof,udof,2] ⊗ D2[1,:]' .+ C[2,udof,udof,3]*AA ⊗ Diagonal(1 ./x2)[1,:]'
    # boundary traction operators bottom
    Bkb = C[2,udof,udof,1] ⊗ I(N)[N,:]'
    B0b = C[2,udof,udof,2] ⊗ D2[N,:]' .+ C[2,udof,udof,3]*AA ⊗ Diagonal(1 ./x2)[N,:]'

    # top
    Lkk[bc[1],:] .= 0
    Lk[bc[1],:] = Bkt
    L0[bc[1],:] = B0t
    M[bc[1],:] .= 0
    # bottom
    Lkk[bc[2],:] .= 0
    Lk[bc[2],:] = Bkb
    L0[bc[2],:] = B0b
    M[bc[2],:] .= 0

    local temp = ω[i] * hdef / sqrt(λs/ρs) 
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

# remove negative real(k) and purely imaginary solutions
f = repeat(ω, 1, Int(2 * length(udof) * N)) / (2 * pi)
k[(0.49*pi .< angle.(k)) .| (angle.(k) .< -0.49*pi)] .= NaN 
k[real(k) .> 2e3] .= NaN
f[isnan.(k)] .= NaN

## Plots
kplot = k[:]
idxplot = idx[:]
fplot = f[:]
filter!(!isnan,kplot)
filter!(!isnan,fplot)
shading  = 1 .- abs.(imag.(kplot)) / (0.03*maximum(abs.(imag.(kplot))))

with_theme(theme_latexfonts()) do 
    x = 1:1000
    fig = Figure(size = (1.2*360, 360))
        ax = Axis(fig[1,1])
        ax.limits = (0, 1100, 0, 400)
        ax.xlabel = L"k \,\, \mathrm{(1/m)}"
        ax.ylabel = L"f"
        scatter!(ax, real(kplot), fplot, 
            color = tuple.(:red, shading))
        lines!(x, sqrt(3*μs/ρs) * x / (2*pi), color = :black, 
            linestyle = :dash, label = L"\sqrt{E/ρ_s}" )

    display(fig)
end