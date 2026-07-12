# Atlas-Het v2

Atlas-Het Coordinate-only preserves the original generator and tests coordinate reparameterization invariance. It does not claim to test changing intrinsic curvature.

Mixed-metric samples a smooth height surface containing bowl-like positive-curvature, nearly flat, and saddle-like negative-curvature regions. Its graph is built from midpoint first-fundamental-form lengths. Boundary-stress uses the same surface while increasing chart overlap and cross-chart relation edges to test message transport near chart boundaries.

For a height field `f(x,y)`, the generator computes `g = I + grad(f) grad(f)^T` and Gaussian curvature `det(Hessian(f)) / (1 + ||grad(f)||^2)^2` with autograd. Local chart metrics are obtained by the coordinate pullback `g_u = J_{z<-u}^T g_z J_{z<-u}`. Ground-truth geometry is diagnostic-only and is never appended to model features.

Metric recovery compares reparameterization-invariant edge lengths obtained by pushing local coordinate differences into observation tangent space. Transition error reconstructs observation tangent probes through overlapping charts. Local and cross-chart distortion are symmetric log errors between true and predicted weighted-graph shortest paths.
