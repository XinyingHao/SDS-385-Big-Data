############################################################
####         Exercise 1                                  ###
####                  Preliminiaries                     ###
############################################################

###############  Part I  Linear Regression  ################
library(microbenchmark)
library(Matrix)

set.seed(1234)
# Generate Simulated Data
N <- 2000
P <- 500
beta <- rnorm(P)
eps <- rnorm(N)

X <- matrix(rnorm(N*P), nrow=N)
y <- X %*% beta + eps
W <- diag(1,N)


# Inversion Method
inver_method <- function(y,X,W) {
  sW <- sqrt(W)
  WX <- sW %*% X
  Wy <- sW %*% y
  XWX <- crossprod(WX)
  XWy <- crossprod(WX, Wy)
  betahat <- solve(XWX, XWy)
  return(betahat)
}


# Choloesky Decomposition Method
chol_method <- function(y,X,W) {
  sW <- sqrt(W)
  WX <- sW %*% X
  Wy <- sW %*% y
  XWX <- crossprod(WX)
  XWy <- crossprod(WX, Wy)
  R <- chol(XWX)
  u <- forwardsolve(t(R), XWy)
  betahat <- backsolve(R, u)
  return(betahat)
}


# QR Method
qr_method <- function(y,X,W) {
  sW <- sqrt(W)
  WX <- sW %*% X
  Wy <- sW %*% y
  qr <- qr(WX)$qr
  qty <- qr.qty(qr(WX),Wy)
  betahat <- backsolve(qr,qty)
  return(betahat)
}

# Compare Timing 
microbenchmark(
  inver_method(y,X,W),
  chol_method(y,X,W),
  qr_method(y,X,W),
  times=5)



### Sparsity ###
# Generate Highly Sparse Data
N <- 2000
P <- 500
sparse <- 0.05 #Set Sparsity = 0.05
beta <- rnorm(P)
eps <- rnorm(N)


X <- matrix(rnorm(N*P), nrow=N)
mask <- matrix(rbinom(N*P,1,sparse), nrow=N) 
X <- mask*X
X[1:10,1:10] # quick visual check
Xs <- Matrix(X)
y <- X %*% beta + eps

# Sparse Method
sparse_method <- function(y,X,W) {
  sW <- sqrt(W)
  WX <- sW %*% X
  Wy <- sW %*% y
  XWX <- crossprod(WX)
  XWy <- crossprod(WX, Wy)
  betahat <- Matrix::solve(XWX, XWy)
  return(betahat)
}

# Compare Timing 
microbenchmark(
  inver_method(y,X,W),
  chol_method(y,X,W),
  qr_method(y,X,W),
  sparse_method(y, Xs,W),
  times=5)
