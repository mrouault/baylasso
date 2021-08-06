library(coda)

leffectiveSize <- function(beta){
  ess = effectiveSize(as.mcmc(beta))
  return(ess)
}
