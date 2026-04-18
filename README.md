# xopt

## Optimization in R

Use base R's `optim()` for multivariate optimization:

```r
objective <- function(x) (x[1] - 2)^2 + (x[2] + 1)^2
result <- optim(c(0, 0), objective, method = "BFGS")
result$par
```
