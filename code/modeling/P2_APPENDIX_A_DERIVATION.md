# Appendix A — Hand derivation of the P2 natural-gradient update

For one observation, let the negative log-likelihood be

\[
S(\theta)=-\log p(y\mid\theta).
\]

A natural gradient measures the update using the local geometry of the probability family. For an infinitesimal parameter displacement \(\delta\), the KL divergence is locally

\[
D_{KL}(p_\theta\Vert p_{\theta+\delta})\approx \tfrac12\delta^T I(\theta)\delta,
\]

where \(I(\theta)\) is the Fisher information. The steepest first-order reduction in score under a fixed distribution-space step solves

\[
\min_\delta\; \nabla S(\theta)^T\delta
\quad\text{s.t.}\quad
\delta^T I(\theta)\delta=\varepsilon.
\]

Using a Lagrange multiplier gives

\[
\delta\propto-I(\theta)^{-1}\nabla S(\theta),
\]

so the natural gradient is

\[
\widetilde\nabla S(\theta)=I(\theta)^{-1}\nabla S(\theta).
\]

At boosting round \(b\), the implementation therefore: (1) computes the score gradient for every observation, (2) solves the Fisher system for the natural gradient, (3) fits one regression tree per distribution parameter to that vector target, (4) chooses a scalar line-search value \(\rho_b\) by minimizing NLL, and (5) applies

\[
\theta_i^{(b)}=\theta_i^{(b-1)}-\eta\rho_b f_b(x_i).
\]

## A.1 Distribution initialization and tree targets

Before boosting, a single marginal distribution is fitted to the training responses. Its fitted parameter vector \(\theta^{(0)}\) is copied to every training observation. For a univariate Normal this is the empirical marginal mean and log standard deviation. For the multivariate Normal it is the empirical mean together with the lower-triangular Cholesky factor of the empirical precision matrix, with diagonal entries represented on the log scale.

For \(n\) observations and \(q\) free distribution parameters, the score layer returns the matrix

\[
G=\begin{bmatrix}
\widetilde\nabla S_1^T\\
\vdots\\
\widetilde\nabla S_n^T
\end{bmatrix}\in\mathbb R^{n\times q}.
\]

The implementation fits **one regression tree for each column of \(G\)**. Thus the target of tree \(j\) is the \(j\)-th natural-gradient coordinate across all sampled training observations. The collection of tree predictions forms \(f_b(x)\in\mathbb R^q\), which approximates the natural-gradient field as a function of match features.

## A.2 Line search

Let \(R_b\) denote the matrix of tree predictions at round \(b\). For a candidate scalar \(\rho>0\), the proposed parameter vector is

\[
\theta_i(\rho)=\theta_i^{(b-1)}-\rho R_{b,i}.
\]

The corrected P2 procedure chooses the scale using the negative log-likelihood objective,

\[
\rho_b\approx\arg\min_{\rho>0}\sum_i S\!\left(y_i;\theta_i^{(b-1)}-\rho R_{b,i}\right).
\]

The implementation uses a discrete doubling/backtracking search: it expands the step while NLL improves, then halves the step until it obtains a finite lower score (or the proposed movement is below tolerance). The learning-rate parameter \(\eta\) is applied after this scalar scale is selected.

## A.3 Univariate Normal

For \(Y\sim\mathcal N(\mu,\sigma^2)\), use parameters \((\mu,s)\) with \(s=\log\sigma\). Ignoring constants,

\[
S(\mu,s)=s+\frac{(y-\mu)^2}{2e^{2s}}.
\]

Thus

\[
\frac{\partial S}{\partial\mu}=\frac{\mu-y}{\sigma^2},
\qquad
\frac{\partial S}{\partial s}=1-\frac{(y-\mu)^2}{\sigma^2}.
\]

The Fisher information is

\[
I(\mu,s)=\begin{bmatrix}1/\sigma^2&0\\0&2\end{bmatrix},
\]

hence

\[
\widetilde\nabla S=
\begin{bmatrix}
\mu-y\\
\tfrac12\left(1-\frac{(y-\mu)^2}{\sigma^2}\right)
\end{bmatrix}.
\]

This is the location/log-scale update used by the project Normal implementation.

## A.4 Categorical distribution

For \(K\) classes, the implementation uses class 0 as the zero-logit reference and stores \(K-1\) free logits. With probabilities \(p\) and one-hot observation \(o\), the free-logit gradient is

\[
\nabla S = p_{1:K}-o_{1:K}.
\]

The Fisher matrix is the covariance of the corresponding categorical indicators,

\[
I=\operatorname{diag}(p_{1:K})-p_{1:K}p_{1:K}^T.
\]

The code solves \(I\widetilde\nabla S=\nabla S\) in batch for the observations supplied to the score layer, and the resulting natural-gradient columns become the regression-tree targets.

## A.5 Multivariate Gaussian score derivatives

For the P2 experiment,

\[
Y\mid X=x\sim\mathcal N_P(\mu,\Sigma).
\]

The implementation parameterizes the precision as

\[
\Sigma^{-1}=LL^T,
\]

where \(L\) is lower triangular and each diagonal element is exponentiated, guaranteeing positive diagonal entries and therefore a positive-definite covariance. With \(d=\mu-y\) and \(\eta=L^Td\), the log density can be written

\[
\log p(y\mid\mu,L)=-\frac{P}{2}\log(2\pi)+\sum_i\log L_{ii}-\frac12\eta^T\eta.
\]

Therefore the NLL is

\[
S=\frac{P}{2}\log(2\pi)-\sum_i\log L_{ii}+\frac12\eta^T\eta.
\]

The mean gradient follows directly:

\[
\nabla_\mu S=LL^T(\mu-y)=\Sigma^{-1}(\mu-y).
\]

For an off-diagonal free parameter \(L_{ij}\), \(i>j\), differentiating \(\frac12\|L^Td\|^2\) gives

\[
\frac{\partial S}{\partial L_{ij}}=d_i\eta_j.
\]

For a diagonal parameter \(a_i=\log L_{ii}\), the chain rule gives

\[
\frac{\partial S}{\partial a_i}=L_{ii}d_i\eta_i-1.
\]

## A.6 Multivariate Fisher-information structure

Write the parameter vector as \(\theta=(\mu,\lambda)\), where \(\lambda\) contains the free lower-triangular precision-factor parameters in NumPy lower-triangle order and diagonal coordinates are stored as log-diagonals. The implementation uses the block Fisher matrix

\[
I(\theta)=
\begin{bmatrix}
LL^T & 0\\
0 & I_{\lambda\lambda}
\end{bmatrix}.
\]

The location block is therefore exactly the precision matrix. The covariance-parameter block is obtained from the expected Hessian. Define

\[
C=L^T\Sigma.
\]

For a diagonal log-parameter associated with row \(i\), the diagonal Fisher element used by the implementation is

\[
I_{a_i,a_i}=L_{ii}^2\Sigma_{ii}+C_{ii}L_{ii}.
\]

For a diagonal coordinate \(a_i\) and an off-diagonal coordinate \(L_{q i}\) sharing the same column \(i\),

\[
I_{a_i,L_{q i}}=\Sigma_{q i}L_{ii}.
\]

For two off-diagonal coordinates whose column indices agree, the implementation uses the corresponding covariance element. In its index notation, for \(L_{j i}\) and \(L_{k i}\),

\[
I_{L_{j i},L_{k i}}=\Sigma_{k j}.
\]

All remaining covariance-parameter entries retain their identity initialization unless one of these analytic cases applies. The matrix is constructed for every observation and the batched linear system

\[
I_i\widetilde\nabla S_i=\nabla S_i
\]

is solved to produce the per-observation tree targets.

### Explicit two-dimensional ordering

For the football bivariate experiment \(P=2\), NumPy lower-triangle order gives

\[
\theta=(\mu_h,\mu_a,a_{11},L_{21},a_{22}),
\]

with

\[
L=\begin{bmatrix}e^{a_{11}}&0\\L_{21}&e^{a_{22}}\end{bmatrix}.
\]

Hence each boosting round fits five trees: two for the conditional means and three for the precision-factor coordinates. The first \(2\times2\) Fisher block is \(LL^T\); the remaining \(3\times3\) block is filled by the expected-Hessian formulas above.

## A.7 Small two-dimensional worked example

Take

\[
\mu=(1.2,0.8)^T,\qquad
y=(2,0)^T,\qquad
L=\begin{bmatrix}1&0\\0.2&1.1\end{bmatrix}.
\]

Then

\[
d=\mu-y=(-0.8,0.8)^T,
\]

and

\[
\eta=L^Td=
\begin{bmatrix}1&0.2\\0&1.1\end{bmatrix}
\begin{bmatrix}-0.8\\0.8\end{bmatrix}
=
\begin{bmatrix}-0.64\\0.88\end{bmatrix}.
\]

The mean-score gradient is

\[
LL^Td=
\begin{bmatrix}1&0.2\\0.2&1.25\end{bmatrix}
\begin{bmatrix}-0.8\\0.8\end{bmatrix}
=
\begin{bmatrix}-0.64\\0.84\end{bmatrix}.
\]

The off-diagonal precision-factor gradient is

\[
\frac{\partial S}{\partial L_{21}}=d_2\eta_1=0.8(-0.64)=-0.512.
\]

For the two log-diagonal parameters,

\[
\frac{\partial S}{\partial a_{11}}=1(-0.8)(-0.64)-1=-0.488,
\]

\[
\frac{\partial S}{\partial a_{22}}=1.1(0.8)(0.88)-1=-0.2256.
\]

These five ordinary score-gradient components are then preconditioned by the corresponding \(5\times5\) Fisher matrix. The resulting five natural-gradient components are the targets fitted by the five regression trees at that boosting round.

This completes the link from the paper's distributional objective to the exact parameter ordering, Fisher solve, tree targets, line search, and update used in the P2 football experiment.
