.xopt_fd_hessian <- function(fn, par, eps = 1e-4) {
  n <- length(par)
  H <- matrix(0, n, n)
  f0 <- fn(par)
  for (i in seq_len(n)) {
    hi <- eps * max(1, abs(par[[i]]))
    xp <- par; xp[[i]] <- par[[i]] + hi
    xm <- par; xm[[i]] <- par[[i]] - hi
    H[[i, i]] <- (fn(xp) - 2 * f0 + fn(xm)) / (hi * hi)
    if (i < n) {
      for (j in (i + 1L):n) {
        hj <- eps * max(1, abs(par[[j]]))
        xpp <- par; xpp[[i]] <- par[[i]] + hi; xpp[[j]] <- par[[j]] + hj
        xpm <- par; xpm[[i]] <- par[[i]] + hi; xpm[[j]] <- par[[j]] - hj
        xmp <- par; xmp[[i]] <- par[[i]] - hi; xmp[[j]] <- par[[j]] + hj
        xmm <- par; xmm[[i]] <- par[[i]] - hi; xmm[[j]] <- par[[j]] - hj
        hij <- (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4 * hi * hj)
        H[[i, j]] <- hij
        H[[j, i]] <- hij
      }
    }
  }
  H
}
