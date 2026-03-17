# Guided wave dispersion in a tube with fluid inside and outside
# Load packages
# Easy identity matrix and kronecker tensor notation
using LinearAlgebra, Kronecker
# Hankel functions for BC
using SpecialFunctions
# Plotting 
using CairoMakie
# POnG
using POnG
BLAS.set_num_threads(1)

## Problem definition: We look for waves propagating in direction 1 in a cylinder with mean radius R with wall thickness h.
#
#                           top, fluid
#  e2 (= r)             ---------------------------------- 
#  |_ e3 (= θ)          - solid: μs, λs, ρs              - 
#                       ---------------------------------- 
#                           bottom, fluid until r = 0

# Waves propagate in direction 1 (= x),  U(x,t) = u(r)e^{i(kx+nθ-ωt)}
# normalization using λf, ρs and h

# displacement dof
udof = 1:3
# Solid properties 
ρs = 1070
μs = 23e3
vL = 1500
λs = vL^2*ρs - 2*μs
R = 4e-3 #radius of curvature
h = 1e-3
n = 1
# Fluid properties 
ρf = 1000
vf = 1500
μf = μs * 1e-12 # We model the fluid as a solid with μ → 0
λf = vf^2 * ρf - 2 * μf

# Define elastic tensor
C = C_elastic(λs, μs)
C = C ./ λf # normalization
# Define elastic "fluid" tensor
Cf = C_elastic(λf, μf)
Cf = Cf ./ λf # normalization

# Discretization in solid
N = 12 # number of nodes in direction 2
x2, D = chebdif(N, 2) # Nodes in [-1, 1]
x2 = (R .+ (h .* x2) ./ 2) ./ h
D2 = 2 .* (h/h) * D[:,:,1]
D22 = 4 .* (h/h)^2 * D[:,:,2]
A = [0 0 0; 0 0 -1; 0 1 0] # Derivation of the basis vectors
AA = (1im * n * I(length(udof)) .+ A) 

# Build the polynomial eigenvalue problem
Lkk, Lk, L0, M = elastodynamics1D_tube(C, x2, D2, D22, AA, udof)

# boundary traction operators top
Bkt = C[2,udof,udof,1] ⊗ I(N)[1,:]'
B0t = C[2,udof,udof,2] ⊗ D2[1,:]' .+ C[2,udof,udof,3]*AA ⊗ Diagonal(1 ./x2)[1,:]'
# boundary traction operators bottom
Bkb = C[2,udof,udof,1] ⊗ I(N)[N,:]'
B0b = C[2,udof,udof,2] ⊗ D2[N,:]' .+ C[2,udof,udof,3]*AA ⊗ Diagonal(1 ./x2)[N,:]'
# displacement bottom 
D0b = I(length(udof)) ⊗ I(N)[N,:]'
# Outer fluid addition
e2 = zeros(length(udof)); e2[2] = 1
Bρft = 1im*ρf/ρs * I(3).*e2 ⊗ I(N)[1,:]'  # approximate BC to get a ploynomial eigenvalue problem

# Discretization in fluid
P = 10
xf2 = lgrpointsleft(P)
Df = lgrdiffleft(P,xf2)
xf2 = (1/2 * (R - h/2) .+ (R - h/2) .* -xf2 ./ 2 ) ./ h
Df2 =  - 2 .* (h/(R - h/2)) * Df # normalization !
Df22 = 4 * (h/(R - h/2))^2 * Df*Df

# Build the polynomial eigenvalue problem
Lfkk, Lfk, Lf0, Mf = elastodynamics1D_tube(Cf, xf2, Df2, Df22, AA, udof)
Mf = Mf * ρf / ρs

# boundary traction operators top
Bfkt = Cf[2,udof,udof,1] ⊗ I(P)[1,:]'
Bf0t = Cf[2,udof,udof,2] ⊗ Df2[1,:]' .+ Cf[2,udof,udof,3]*AA ⊗ Diagonal( 1 ./ xf2)[1,:]'
# displacement top 
Df0t = I(length(udof)) ⊗ I(P)[1,:]'

# Impose solid BC: 
bc =  [(0:length(udof)-1).*N .+ 1, (1:length(udof)).*N]
# top
Lkk[bc[1],:] .= Bkt
Lk[bc[1],:] .= B0t
L0[bc[1],:] .= 0
M[bc[1],:] .= Bρft
# bottom
Lkk[bc[2],:] .= 0
Lk[bc[2],:] .= Bkb
L0[bc[2],:] .= B0b
M[bc[2],:] .= 0

# Impose fluid BCs
bcf =  [(0:length(udof)-1).*P .+ 1, (1:length(udof)).*P]
# top
Lfkk[bcf[1],:] .= 0
Lfk[bcf[1],:] .= 0
Lf0[bcf[1],:] .= Df0t
Mf[bcf[1],:] .= 0

# Assemble matrices
Fkk, Fk, F0, Fd = assemblelayer(Lkk,Lk,L0,M,Lfkk,Lfk,Lf0,Mf,bc,bcf,D0b,Bfkt,Bf0t)

## Solve polynomial eigenvalue problem
# Impose ω and get eigenvalues and eigenvectors
ω = 2 * pi * range(1e-2, 400, length = 70) 
k = Array{ComplexF64,2}(undef, length(ω), 2 * length(udof) * (N + P))
u = Array{ComplexF64,3}(undef, length(ω), length(udof) * (N+P), 2 * length(udof) *(N+P))

@time for i in axes(ω,1)
    temp = ω[i] * h / sqrt(λf/ρs) 
    problem = PEP([temp^2 .* Fd .+ F0, Fk, Fkk])
    ktmp, utmp = polyeig(problem)
    k[i,:] = -1im * ktmp' / h
    u[i,:,:] = utmp
end

# Store result of a 1D calculation before (and for) postprocessing
result = result1D(udof,N,ω,k,u)
# remove negative real(k) and purely imaginary solutions
#k[(0.49*pi .< angle.(k)) .| (angle.(k) .< -0.49*pi)] .= NaN 
# Remove evanescent solutions
k[(1 .- abs.(imag.(k[:]))) .< 0] .= NaN
k[real(k) .> 1500 .|| real(k).< 0] .= NaN
## Plots
kplot = k[:]
fplot = repeat(ω, 1, Int(2 * length(udof) * (N+P)))[:] / (2 * pi)
fplot[isnan.(kplot)] .= NaN
kplot = filter(!isnan,kplot)
fplot = filter(!isnan,fplot)

# Perform an iterative process to apply the accurate BC
# Initialisation of the iterative process
kplot2 = Vector{ComplexF64}()
# Make blocks once to reassemble Fd
TRB = zeros(ComplexF64, size(Lkk)[1], size(Lfkk)[1])
BLB = zeros(ComplexF64, size(Lfkk)[1], size(Lkk)[1])
# Loop on all (ω,k) eigenvalues couples
niter = 15
for i in axes(kplot,1)
    # current (ω,k)
    ktmp = kplot[i]
    ωtmp = fplot[i] * 2*pi
    # find associated eigenvector
    idxf = argmin(abs.(ω .- ωtmp))
    uf = u[idxf,:,:]
    kf = k[idxf,:]
    uf = uf[:,isfinite.(kf)]
    kf = kf[isfinite.(kf)]
    idxk = argmin(abs.(real.(kf) .- ktmp))
    utmp = uf[:,idxk]

    #compute the new BC
    tol = 1
    tic = 1
    while tol > 1e-3 && tic < niter
        tic = tic + 1
        ky = sqrt((ω[idxf] * h / vf)^2 - (ktmp*h)^2)
        if imag(ky) < 0
            ky = conj(ky)
        end
        ro = (R + h/2) / h
        Bρftup =  I(3).*e2 ⊗ I(N)[1,:]' * (1im*ktmp*h) / ky *
            (ρf/ρs * hankelh1(n, ky*ro)/(n/(ky*ro) * hankelh1(n, ky*ro) - hankelh1(n+1, ky*ro))) 
        # update BC
        M[bc[1],:] .= Bρftup
        # reassemble Fd
        tt = hcat(M,TRB)
        tb = hcat(BLB,Mf)
        Fdup = vcat(tt,tb)
        # helper matrices for inverse iteration
        A1 = (1im*ktmp*h)^2*Fkk .- F0 .- (ω[idxf] * h / sqrt(λf/ρs))^2*Fdup
        A2 = 2*1im*ktmp*h*Fkk .+ Fk

        utmp = (A1 .- 1im*ktmp*h*A2) \ A2*utmp
        sol = (A2*utmp \ A1*utmp) * -1im /h 
        tol = abs(real(ktmp) - real(sol))
        ktmp = sol
    end
    push!(kplot2,ktmp)
end

with_theme(theme_latexfonts()) do 
    x = 0:1000
    fig = Figure()
        ax = Axis(fig[1,1])
        ax.limits = (0, 1100, 0, 400)
        ax.xlabel = L"k \; \mathrm{(1/m)}"
        ax.ylabel = L"f \; \mathrm{(Hz)}"
        ax.xlabelsize = 20
        ax.ylabelsize = 20
            scatter!(ax, real(kplot), fplot, marker = :utriangle)
            scatter!(ax, real(kplot2), fplot)
            lines!(ax,x, sqrt(3*μs/ρs) * x / (2*pi), color = :black, linestyle = :dash, linewidth = 3, label = L"\sqrt{E_s/ρ_s}")
            lines!(ax,x, sqrt(μs/ρs) * x / (2*pi), color = :black, linewidth = 3, label = L"\sqrt{μ_s/ρ_s}")
            lines!(ax, x, x .* sqrt(3*μs/ρf * h / R / 2) / (2*pi), linewidth = 3, color = :green, label = L"v_b")
            lines!(ax,x, 1500 * x / (2*pi), linewidth = 3, color = :red, label = L"v_f")
        axislegend(position = :rb)
        ax2 = Axis(fig[1,2])
        ax2.limits = (0, 2, 0, 400)
        ax2.xlabel = L"k \; \mathrm{(1/m)}"
        ax2.ylabel = L"f \; \mathrm{(Hz)}"
        ax2.xlabelsize = 20
        ax2.ylabelsize = 20
            scatter!(ax2, real(kplot), fplot, marker = :utriangle)
            scatter!(ax2, real(kplot2), fplot)
            lines!(ax2,x, sqrt(3*μs/ρs) * x / (2*pi), color = :black, linestyle = :dash, linewidth = 3, label = L"\sqrt{E_s/ρ_s}")
            lines!(ax2,x, sqrt(μs/ρs) * x / (2*pi), color = :black, linewidth = 3, label = L"\sqrt{μ_s/ρ_s}")
            lines!(ax2, x, x .* sqrt(3*μs/ρf * h / R / 2) / (2*pi), linewidth = 3, color = :green, label = L"v_b")
            lines!(ax2,x, 1500 * x / (2*pi), linewidth = 3, color = :red, label = L"v_f")
    resize_to_layout!(fig)
    display(fig)
end

