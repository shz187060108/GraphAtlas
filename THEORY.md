# Corrected coordinate-equivariant GraphAtlas operator

## 1. Why the source-only transition is insufficient

Let node `j` be represented in chart `l`, and node `i` in chart `k`. The source-only transition Jacobian

\[
J\tau_{kl}(u_j^{(l)})
\]

maps a vector at node `j` into chart `k` at the same base point `j`. A target self-vector is based at `i`. Under a nonlinear reparameterization `rho_k`, these vectors transform with `J rho_k(u_j)` and `J rho_k(u_i)`. They therefore cannot be added in a coordinate-equivariant manner unless the reparameterization is globally affine or an additional within-chart connection transport is supplied.

## 2. Encoder-decoder induced point-to-point transport

Let the shared observation representation be `h_i` and define local charts

\[
u_i^{(k)}=\phi_k(h_i),\qquad \widehat h_i^{(k)}=\psi_k(u_i^{(k)}).
\]

For a tangent vector `a_j^(l)` at the source, define

\[
T_{j,l\to i,k}a_j^{(l)}
=
J\phi_k(h_i)J\psi_l(u_j^{(l)})a_j^{(l)}.
\]

This is an encoder-decoder induced pull-push transport. At the same node it reduces to the Jacobian of the induced transition map:

\[
T_{i,l\to i,k}=J(\phi_k\circ\psi_l)(u_i^{(l)}).
\]

## 3. Reparameterization law

For independent smooth diffeomorphisms `rho_k`, set

\[
\widetilde\phi_k=\rho_k\circ\phi_k,
\qquad
\widetilde\psi_k=\psi_k\circ\rho_k^{-1}.
\]

Write

\[
R_k(i)=J\rho_k(u_i^{(k)}).
\]

Then

\[
\widetilde T_{j,l\to i,k}
=
R_k(i)T_{j,l\to i,k}R_l(j)^{-1}.
\]

Thus, if `a_j^(l)` transforms as `R_l(j)a_j^(l)`, the transported vector transforms as a target tangent vector:

\[
\widetilde T_{j,l\to i,k}\widetilde a_j^{(l)}
=
R_k(i)T_{j,l\to i,k}a_j^{(l)}.
\]

## 4. Permitted learnable operations

Store `C` vector channels in a matrix

\[
A_i^{(k)}\in\mathbb R^{d\times C}.
\]

Coordinate changes act from the left. Therefore learnable channel mixing may act from the right:

\[
A_i^{(k)}W,
\qquad W\in\mathbb R^{C\times C}.
\]

This commutes with coordinate changes. A generic matrix acting on the coordinate dimension does not.

The preactivation is

\[
B_i^{(k)}
=
A_i^{(k)}W_{\rm self}
+
\sum_{j\in\mathcal N(i)}\sum_l
\alpha_{ij}^{kl}q_{jl}
T_{j,l\to i,k}
A_j^{(l)}W_{\rm msg}.
\]

The coefficients `alpha` are scalars computed only from chart-invariant quantities, such as observation features, graph attributes, and norms after decoder pushforward.

## 5. Equivariant nonlinearity

Coordinate-wise ReLU or GELU is not equivariant under arbitrary smooth reparameterizations. GraphAtlas instead uses invariant scalar gates. Push each vector channel into observation space:

\[
S_i^{(k)}=J\psi_k(u_i^{(k)})B_i^{(k)}.
\]

The channel norms and Gram matrix

\[
\|S_{i,c}^{(k)}\|_2,
\qquad
(S_i^{(k)})^\top S_i^{(k)}
\]

are unchanged by chart reparameterization. An MLP maps these invariants to channel gates `gamma_i^(k)`. The update is

\[
A_i^{(k)\prime}
=
B_i^{(k)}\operatorname{Diag}(\gamma_i^{(k)})
+
A_i^{(k)}W_{\rm res}.
\]

Because every gate is scalar and invariant, this update is equivariant.

## 6. Invariant readout

Push the final chart vectors to observation space and combine them there:

\[
R_i
=
\sum_k q_{ik}J\psi_k(u_i^{(k)})A_i^{(k)}.
\]

Each summand is invariant under a local chart reparameterization. Any task head applied to `R_i` and `h_i` therefore has identical predictions before and after independent coordinate changes.

## 7. Theorem

Assume:

1. every `rho_k` is a smooth local diffeomorphism;
2. chart memberships and scalar attention coefficients are unchanged by reparameterization;
3. coordinate-axis linear maps and coordinate-wise nonlinearities are not used;
4. all vector-channel mixing acts from the right;
5. all nonlinear gates depend only on invariant scalars.

Then every GraphAtlas layer is equivariant:

\[
\widetilde A_i^{(k,t)}=R_k(i)A_i^{(k,t)}.
\]

The observation readout and task predictions are invariant:

\[
\widetilde R_i=R_i,
\qquad
\widetilde{\widehat y}_i=\widehat y_i.
\]

The proof is by induction over layers using the transport transformation law, commutation of right channel mixing, invariance of scalar gates, and cancellation between `J psi_k R_k^{-1}` and `R_k A_i` in the final pushforward.

## 8. Metric and consistency losses

The conceptual pullback metric is

\[
g_k=J\psi_k^\top J\psi_k.
\]

No coordinate-dependent `epsilon I` term is part of the mathematical definition. Numerical linear algebra may use a small solve jitter, reported as an implementation tolerance.

Metric compatibility is implemented through random tangent probes in observation space:

\[
J\psi_k v
\quad\text{versus}\quad
J\psi_l J\phi_l J\psi_k v.
\]

Transition consistency has three diagnostics. On every pairwise overlap, GraphAtlas penalizes a decoded inverse-cycle defect. It also constructs a differentiable chart-overlap graph from soft memberships. For every supported two-hop route \(k\to l\to m\), it compares the direct and composed transitions after decoding into observation space:

\[
\left\|
\psi_m(\tau_{mk}(u))
-
\psi_m(\tau_{ml}(\tau_{lk}(u)))
\right\|_2^2.
\]

The route weight is the product of the two soft overlap-graph edges and the endpoint memberships. This path-consistency term remains non-vacuous when `membership_topk=2`, where no node belongs to three charts simultaneously. On genuine triple overlaps, the implementation additionally reports the classical triple-overlap cocycle defect. All comparisons are made in observation space rather than by coordinate-dependent Euclidean differences between chart coordinates, so the losses are unchanged by smooth chart reparameterizations.

## 9. Why point-to-point transport is well defined

The observation representation lives in the fixed vector space \(\mathbb R^p\). Its tangent spaces are canonically identified with \(\mathbb R^p\) at every base point. The composition

\[
J\phi_k(h_i)J\psi_l(u_j^{(l)})
\]

therefore uses the ambient Euclidean connection to move the decoder-pushed source vector between observation-space base points before pulling it into the target chart. No identification between tangent spaces of two arbitrary curved manifolds is assumed. If the shared observation space were itself a learned curved manifold, an explicit connection or parallel transport in that space would be required.

## 10. Layerwise proof details

Assume the induction hypothesis

\[
\widetilde A_j^{(l,t)}=R_l(j)A_j^{(l,t)}.
\]

The channel-mixed source state obeys

\[
\widetilde A_j^{(l,t)}W_{\rm msg}
=
R_l(j)A_j^{(l,t)}W_{\rm msg}.
\]

Using the transport law gives

\[
\widetilde T_{j,l\to i,k}
\widetilde A_j^{(l,t)}W_{\rm msg}
=
R_k(i)T_{j,l\to i,k}A_j^{(l,t)}W_{\rm msg}.
\]

Because memberships and attentional coefficients are invariant scalars, every transported message and their sum transform by the same left factor \(R_k(i)\). The self term has the same transformation law because channel mixing acts on the right. Thus

\[
\widetilde B_i^{(k)}=R_k(i)B_i^{(k)}.
\]

For the decoder pushforward,

\[
J\widetilde\psi_k(\widetilde u_i^{(k)})
=
J\psi_k(u_i^{(k)})R_k(i)^{-1}.
\]

Hence

\[
J\widetilde\psi_k\widetilde B_i^{(k)}
=
J\psi_k B_i^{(k)},
\]

so every norm, Gram entry and scalar gate computed from the pushed vectors is unchanged. Multiplication by invariant channel gates and the equivariant residual term therefore yields

\[
\widetilde A_i^{(k,t+1)}=R_k(i)A_i^{(k,t+1)}.
\]

The base case holds because the initial observation-space vectors are pulled through \(J\phi_k\). Finally,

\[
J\widetilde\psi_k\widetilde A_i^{(k,T)}
=
J\psi_k A_i^{(k,T)},
\]

and the membership-weighted observation-space readout is invariant.

## 11. Scope of the theorem

The theorem concerns reparameterizations of learned chart coordinates. It does not assert invariance to arbitrary changes of the input feature basis, graph rewiring, or changes in chart membership. It also does not imply that a learned chart atlas is identifiable. Multiple atlases can yield the same invariant predictor. The reconstruction, cocycle, metric and geometry terms constrain this non-identifiability but do not remove it completely.
