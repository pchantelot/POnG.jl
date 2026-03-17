# Examples
We illustrate the usage of the POnG package by computing the dispersion relation of guided elastic waves in two distinct cases:
- a soft hydrogel layer sensitive to surface tension,
- a viscoelastic elastomer plate subjected to uniaxial tension,

that showcase the ability to consider **varied boundary conditions** and to include the effects of **viscoelasticity** and **hyperelasticity**.

In both cases, the dispersion relation is obtained by finding non-trivial solutions of the elastodynamics equation
```math
    \boldsymbol{\nabla} \cdot (\boldsymbol{C} : \boldsymbol{\nabla} \boldsymbol{u}) + \rho_s \omega^2\boldsymbol{u} = 0, \tag{1}
```
where $\boldsymbol{C}$ is the elastic tensor, $\rho_s$ is the density, and $\boldsymbol{u} = \boldsymbol{u}e^{i(k_x x-\omega t)}$ is the displacement vector, subjected to appropriate boundary conditions.

The package relies on a semi-analytical method that transforms the initial boundary value problem into a polynomial eigenvalue problem by discretizing the waveguide's cross-section. 

To clarify the relationship between equation $(1)$ and the boundary conditions with the code presented in this tutorial, we provide the following expressions that derive from writing the nabla operator as 
```math
\boldsymbol{\nabla} = ik_x \boldsymbol{e}_x + \partial_y \boldsymbol{e}_y.
```
Equation $(1)$ then reads
```math
\left[(ik_x)^2 \boldsymbol{c}_{xx} + ik_x \left(\boldsymbol{c}_{xy} + \boldsymbol{c}_{yx}\right)\partial_y + \boldsymbol{c}_{yy}\partial_y^2 + \omega^2\rho_s \right] \cdot \boldsymbol{u} = \boldsymbol{0}, \tag{2}
```
where $\boldsymbol{c}_{ij} = \boldsymbol{e}_i \cdot \boldsymbol{C} \cdot \boldsymbol{e}_j$, and the normal traction at the waveguide's interface reads
```math
\boldsymbol{e}_y \cdot \boldsymbol{C} : \boldsymbol{\nabla} \boldsymbol{u} = \left[ik_x\boldsymbol{c}_{yx} + \boldsymbol{c}_{yy}\partial_y\right]\cdot \boldsymbol{u}. \tag{3}
```

!!! tip 
    For details on the implementation see the [Computing guided waves](@ref) section and [Kiefer, 2022](http://doi.org/10.25593/978-3-96147-550-6).

## Waves in a soft hydrogel layer
We first load the POnG package and two packages necessary to perform linear algebra operations.
```julia
using LinearAlgebra, Kronecker
using POnG
```
Next, we define the material and geometrical parameters,
```julia
μs = 380 # shear modulus
ρs = 1000 # density
λs = 1.2e7 # first Lamé parameter
γ = 0.07 # surface tension
h = 9.8e-3 # layer thickness
```
and we compute the fourth order elastic tensor. 
```julia
C = C_elastic(λs, μs)
C = C ./ μs # normalization
```
Note that we will use the shear modulus, the density and the thickness to normalize the problem.

Then, we prepare the discretization of the problem by computing the collocation points and the spectral differentiation matrices,
```julia
N = 12 # number of collocation points
_, D = chebdif(N,2)
D2 = .- 2 * (h/h) * D[:,:,1] # ∂_y
D22 = 4 * (h/h) * D[:,:,2] # ∂_yy
```
enabling us to obtain the discretized version of equation $(2)$ 
```math 
\left[(ik_x)^2 L_{kk} + ik_x L_k + L_0 + \omega^2M \right] u = 0.
```
```julia
udof = 1:3 # We consider displacements in directions 1, 2 and 3 (corresponding to x, y, and z)
Lkk, Lk, L0, M = elastodynamics1D(C, D2, D22, udof)
```
We now impose the boundary conditions by replacing the elastodynamics equations at the boundary collocation points (points $1$ and $N$) by the boundary conditions.  At the bottom interface, the layer is attached to the wall and we impose zero displacement, $\boldsymbol{I}\cdot\boldsymbol{u} = 0$.
```julia
#Location of boundary vectors in discretized matrix
bc = [(0:length(udof)-1).*N .+ 1, (1:length(udof)).*N] 
Lkk[bc[2],:] .= 0 # 2 corresponds to point N (at the bottom)
Lk[bc[2],:] .= 0
L0[bc[2],:] .= I(length(udof)) ⊗ I(N)[N,:]'
M[bc[2],:] .= 0
```
At the top interface, in the presence of surface tension, the normal traction (equation $(3)$) reads $\boldsymbol{e}_y \cdot \boldsymbol{C} : \boldsymbol{\nabla} \boldsymbol{u} = \left[ik_x\boldsymbol{c}_{yx} + \boldsymbol{c}_{yy}\partial_y\right]\cdot \boldsymbol{u} \approx - \gamma k_x^2 u_y\boldsymbol{e}_y$.
```julia
e2 = zeros(length(udof)); e2[2] = 1 # interface normal
Lkk[bc[1],:] .=  γ/(μs*h) .* I(length(udof)) .* e2 ⊗ I(N)[1,:]' # -γ/(μs*h)(ik_x)^2e_y
Lk[bc[1],:] .= C[2,udof,udof,1] ⊗ I(N)[1,:]' # c_{yx}
L0[bc[1],:] .= C[2,udof,udof,2] ⊗ D2[1,:]' # = c_{yy}∂y   
M[bc[1],:] .= 0
```
Note that we introduced the boundary conditions by putting each contribution in the corresponding matrices $L_0$, $L_k$, and $L_{kk}$ according to the order of the eigenvalue $ik_x$. 

We can now solve the polynomial eigenvalue problem. For each angular frequency of interest $\omega$ we solve for the eigenpair ($k_x$, $u$), enabling us to extract a complex-valued wavenumber. 
```julia
ω = 2*pi .* range(1e-2,150, length = 100)
k = Array{ComplexF64,2}(undef, length(ω), 2 * length(udof) * N)
u = Array{ComplexF64,3}(undef, length(ω), length(udof) * N, 2 * length(udof) * N)
# create and solve polynomial eigenvalue problem
@time for i in eachindex(ω)
    local temp = ω[i] * h / sqrt(μs/ρs)
    local problem = PEP([temp^2 .* M .+ L0, Lk, Lkk])
    local ktmp, utmp = polyeig(problem)
    k[i,:] = -1im * ktmp' / h
    u[i,:,:] = utmp
end
```
## Incremental waves in a viscoelastic plate in simple tension
We first load the POnG package and two packages necessary to perform linear algebra operations.
```julia
using LinearAlgebra, Kronecker
using POnG
```
Next, we define the material and geometrical parameters,
```julia
# Ecoflex OO-30 properties
μs = 23e3 # shear modulus
α = 0.29 # Mooney-Rivlin model parameter C1 = μsα, C2 = μs*(1-α)
ρs = 1070 # density
vL = 1000 # longitudinal wave velocity
h = 1e-3 # plate thickness
τ = 330e-6 # relaxation time for the Kelvin-Voigt model
n = 0.32 # fractional derivative exponent
```
and the characteristics of the underlying uniaxial deformation.
```julia
# Stretch ratio in directions 1 and 3
λ1 = 1.2
λ3 = 1 / sqrt(λ1)
hdef = h / (λ1*λ3) # deformed thickness
β = 0.29 # Coupling between viscous effects and pre-stress
```
!!! tip 
    For details on the modelling of viscous effects and the calculation of the push-forward of the elastic tensor see the [Viscoacoustoelasticity](@ref) section and the [GEW soft strip](https://github.com/dakiefer/GEW_soft_strip) repository.

We prepare the discretization of the problem by computing the collocation points and the spectral differentiation matrices.
```julia
N = 12 # number of collocation points
_, D = chebdif(N,2)
D2 = .- 2 * (hdef/hdef) * D[:,:,1] # ∂_y
D22 = 4 * (hdef/hdef) * D[:,:,2] # ∂_yy
```
Unlike in the previous example the modified elasticity tensor and the polynomial eigenvalue problem are frequency dependent, we thus calculate them for each pulsation at which we want to obtain the wavenumber.
```julia
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
```
!!! note
    In this example the elasticity tensor is computed using the function `C_MR` corresponding to the Mooney-Rivlin hyperelastic model. We provide many hyperelastic models as detailed in section [Viscoacoustoelasticity](@ref).

!!! note
    Here, we consider a plate in air so that the boundary condition at both interfaces is of zero normal traction, $\boldsymbol{e}_y \cdot \boldsymbol{C} : \boldsymbol{\nabla} \boldsymbol{u} = \left[ik_x\boldsymbol{c}_{yx} + \boldsymbol{c}_{yy}\partial_y\right]\cdot \boldsymbol{u} =0$.


