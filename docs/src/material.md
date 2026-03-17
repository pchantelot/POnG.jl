# Viscoacoustoelasticity
This package provides the tools to compute the dispersion relation of small amplitude waves in homogeneously pre-stressed viscoelastic media. 
In the framework of the acoustoelatic theory, the propagation of incremental waves on top of a finitely deformed configuration is still described by a wave equation where the elasticity tensor $\boldsymbol{C}$ is replaced by a modified elasticity tensor $\boldsymbol{C^0}$, which takes into account the intertwined contributions of pre-stress and viscoleasticity, and where $\boldsymbol{u}$ now stands for the incremental displacement.

This similarity allows to use the methods introduced above for elastic waveguides to compute the dispersion relation in pre-stressed viscoelastic media after:

- computing the modified elasticity tensor $\boldsymbol{C^0}$ 
- adapting the geometry to the considered level of pre-stress, for example $w = \lambda_\theta w_0$, and $h = h_0/(\lambda_\theta \lambda_x)$.

We include functions to obtain the 4-th order incremental $\boldsymbol{C^0}$ for nearly incompressible hyperelastic materials. 
$\boldsymbol{C^0}$ is computed from a chosen strain energy density function $W$, built on the principal invariants of the left Cauchy-Green tensor $\boldsymbol{B} = \boldsymbol{F}\cdot\boldsymbol{F}^T$,
```math
\begin{align*}
    &I_1 = tr(\boldsymbol{B}) = \lambda_1^2 + \lambda_2^2 +\lambda_3^2, \\
    &I_2 = \frac{1}{2}\left(\mathrm{Tr}(\boldsymbol{B})^2 - \mathrm{Tr}(\boldsymbol{B}^2)\right) = \lambda_2^2\lambda_3^2 + \lambda_1^2\lambda_3^2 + \lambda_1^2\lambda_2^2, \\
    &I_3 = \mathrm{det}(\boldsymbol{B}) = \lambda_1^2\lambda_2^2\lambda_3^2 = J^2.
\end{align*}
```
The coefficients of the push-forward of $\boldsymbol{C}$ in the deformed configuration can be found in [Delory *et al.*, 2024](https://doi.org/10.1039/D3SM01485A) and are identical to that given in  [Ogden, 1997](https://scholar.google.com/citations?hl=fr&user=vEq5LKUAAAAJ) with a permutation of the last two indices.
The effect of viscoelasticity is incorporated in the modified elasticity tensor $\boldsymbol{C^0}$, by adding a term to the push-forward of $\boldsymbol{C}$ based on the fractional Kelvin-Voigt model.
$\boldsymbol{C^0}$ becomes frequency dependent and reads
```math
\begin{equation*}
    C^0_{jikl}(\lambda_1,\lambda_2,\lambda_3, \omega) = C^0_{jikl}(\lambda_1,\lambda_2,\lambda_3)+\mu_0I_{jikl}\left(1 + \beta'\frac{\lambda_i^2+\lambda_j^2-2}{2}\right)(\mathrm{i}\omega\tau)^n,
\end{equation*}
```
with $I_{jikl} = (\delta_{jk}\delta_{il}+\delta_{jl}\delta_{ik})$ as detailed in [Delory *et al.*, 2023](https://doi.org/10.1016/j.eml.2023.102018).
The second term, that involves both the principal stretches and the frequency, underlines the coupling between pre-stress and viscoelasticity.

The following constitutive relations have been implemented where we further remove the dependence on the stretch ratio $\lambda_2$ by taking the nearly incompressible limit $J \to 1$ in combination with the boundary condition $\sigma_2 = 0$.
!!! tip
    If you want to implement a different elasticity tensor (*i.e.* different hyperelastic model, viscoelastic model) have a look at the Mathematica files provided in the [GEW soft strip](https://github.com/dakiefer/GEW_soft_strip) repository. 

## Models for the small to moderate strain regime
```@docs
C_MR
C_GT
C_C
```
## Models for the strain-hardening regime
```@docs
C_MRSH
C_MRSH2
C_GTSH
C_GTSH2
C_CSH
```
## Models for the limiting-chain regime
```@docs
C_GMR
C_GG
C_GC
C_DCMR2
C_DCGT
```