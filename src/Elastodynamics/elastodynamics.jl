"""
    elastodynamics1D(C,D2,D22,udof)

Discretize the elastodynamics equation in 1D given the elasticity tensor `C`, the derivation matrices `D2` and `D22`, and the displacement degrees of freedom `udof`.
Returns the four matrices to build the polynomial eigenvalue problem (ik)^2 `Lkk` + (ik) `Lk` + `L0` + ω^2 `M` = 0.
"""
function elastodynamics1D(C,D2,D22,udof)

    # Material matrices used in the elastodynamic equation
    C11 = C[1,udof,udof,1]
    C12 = C[1,udof,udof,2] 
    C21 = C[2,udof,udof,1]
    C22 = C[2,udof,udof,2]
    M = I(length(udof))

    # Discretized matrices to build the polynomial eigenvalue problem
    # (ik)^2 Lkk + (ik) Lk + L0 + ω^2 M = 0
    Lkk = C11 ⊗ I(size(D2, 1)); Lkk = collect(Lkk)
    Lk = (C12 .+ C21) ⊗ D2; Lk = collect(Lk)
    L0 = C22 ⊗ D22; L0 = collect(L0)
    Md = M ⊗ I(size(D2, 1)); Md = collect(Md)

    return Lkk, Lk, L0, Md

end

"""
    elastodynamics1D_tube(C,x2,D2,D22,AA,udof)

Discretize the elastodynamics equation in 1D for a tube on given the elasticity tensor `C`, the collocation points `x2`, derivation matrices `D2` and `D22`, and the displacement degrees of freedom `udof`.
In polar coordinates, we additionally need the matrix `AA` for the derivation of the basis vectors.
Returns the four matrices to build the polynomial eigenvalue problem (ik)^2 `Lkk` + (ik) `Lk` + `L0` + ω^2 `M` = 0.
"""
function elastodynamics1D_tube(C,x2,D2,D22,AA,udof)

    # Material matrices
    C11 = C[1,udof,udof,1]
    C12 = C[1,udof,udof,2]
    C21 = C[2,udof,udof,1]
    C13 = C[1,udof,udof,3]
    C31 = C[3,udof,udof,1]
    C22 = C[2,udof,udof,2]
    C23 = C[2,udof,udof,3]
    C32 = C[3,udof,udof,2]
    C33 = C[3,udof,udof,3]
    M = I(length(udof))

    # Discretized matrices to build the polynomial eigenvalue problem
    # (ik)^2 Lkk + (ik) Lk + L0 + ω^2 Md =0
    Lkk = C11 ⊗ I(size(D2, 1)); Lkk = collect(convert(Matrix{ComplexF64}, Lkk))
    Lk = (C12 .+ C21) ⊗ D2 .+ (C21 .+ C13*AA .+ AA*C31) ⊗ Diagonal(1 ./ x2) 
    L0 = C22 ⊗ D22 .+ (C22 .+ C23*AA .+ AA*C32) ⊗ (Diagonal(1 ./x2)*D2) .+ AA*C33*AA ⊗ Diagonal(1 ./ x2.^2)
    Md = M ⊗ I(size(D2, 1)); Md = collect(convert(Matrix{ComplexF64}, Md))

    return Lkk, Lk, L0, Md

end

"""
    elastodynamics2D(C,D2d,D3d,D22d,D33d,D23d,udof)

Discretize the elastodynamics equation in 2D given the elasticity tensor `C`, the derivation matrices `D2d`, `D3d`, `D22d`, `D33d`, and `D23d` and the displacement degrees of freedom `udof`.
Returns the four matrices to build the polynomial eigenvalue problem (ik)^2 `Lkk` + (ik) `Lk` + `L0` + ω^2 `M` = 0.
"""
function elastodynamics2D(C,D2d,D3d,D22d,D33d,D23d,udof)

    # Material matrices used in the elastodynamic equation
    C11 = C[1,udof,udof,1]
    C12 = C[1,udof,udof,2] 
    C21 = C[2,udof,udof,1]
    C13 = C[1,udof,udof,3]
    C31 = C[3,udof,udof,1]
    C22 = C[2,udof,udof,2]
    C23 = C[2,udof,udof,3]
    C32 = C[3,udof,udof,2]
    C33 = C[3,udof,udof,3]
    M = collect(I(length(udof)))

    # Discretized matrices to build the polynomial eigenvalue problem
    # (ik)^2 Lkk + (ik) Lk + L0 + ω^2 Md =0
    Lkk = C11 ⊗ I(size(D2d,1)); Lkk = collect(Lkk)
    Lk = (C12 .+ C21) ⊗ D2d .+ (C13 .+ C31) ⊗ D3d
    L0 = C22 ⊗ D22d .+ C33 ⊗ D33d .+ (C23 .+ C32) ⊗ D23d
    Md = M ⊗ I(size(D2d,1)); Md = collect(Md)

    return Lkk, Lk, L0, Md

end

"""
    Celastodynamics2D(C,x2,D2,D3,D22,D33,A,udof)

Discretize the elastodynamics equation in 2D given the elasticity tensor `C`, , the collocation points `x2`, the derivation matrices `D2`, `D3`, `D22`, and  `D33`, and the displacement degrees of freedom `udof`.
In polar coordinates, we additionally need the matrix `A` for the derivation of the basis vectors.
Returns the four matrices to build the polynomial eigenvalue problem (ik)^2 `Lkk` + (ik) `Lk` + `L0` + ω^2 `M` = 0.
"""
function Celastodynamics2D(C,x2,D2,D3,D22,D33,A,udof)

    # Material matrices used in the elastodynamic equation
    C11 = C[1,udof,udof,1]
    C12 = C[1,udof,udof,2] 
    C21 = C[2,udof,udof,1]
    C13 = C[1,udof,udof,3]
    C31 = C[3,udof,udof,1]
    C22 = C[2,udof,udof,2]
    C23 = C[2,udof,udof,3]
    C32 = C[3,udof,udof,2]
    C33 = C[3,udof,udof,3]
    M = collect(I(length(udof)))

    # Discretized matrices to build the polynomial eigenvalue problem
    # (ik)^2 Lkk + (ik) Lk + L0 + ω^2 Md =0
    Lkk = C11 ⊗ I(size(D2,1) * size(D3,1)); Lkk = collect(Lkk)
    Lk = (C12 .+ C21) ⊗ I(size(D3,1)) ⊗ D2 .+ (C21 ⊗ I(size(D3,1)) .+ (C13 .+ C31) ⊗ D3 .+ 
        (C13*A .+ A*C31) ⊗ I(size(D3,1))) ⊗ Diagonal(1 ./ x2)
    L0 = C22 ⊗ I(size(D3,1)) ⊗ D22 .+ (C22 ⊗ I(size(D3,1)) .+ C32 ⊗ D3 .+ A*C32 ⊗ I(size(D3,1))) ⊗ (Diagonal(1 ./ x2)*D2) .+ 
        (C23 ⊗ D3 .+ C23*A ⊗ I(size(D3,1))) ⊗ (D2*Diagonal(1 ./ x2) .+ Diagonal(1 ./ x2.^2)) .+
        (C33 ⊗ D33 .+ (C33*A .+ A*C33) ⊗ D3 .+ A*C33*A ⊗ I(size(D3,1))) ⊗ Diagonal(1 ./ x2.^2)
    Md = M ⊗ I(size(D2,1) * size(D3,1)); Md = collect(Md)

    return Lkk, Lk, L0, Md

end