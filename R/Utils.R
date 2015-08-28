#' Utility functions 
#' 
#'  Generate sequence of artificial binary classification data 
#'  from two multivariate normal distribution  for multitask 
#'  learning. All datasets have the 
#'  same number of examples n and the same dimension  d   
#' 
#' @name Utils
#' @param form formula
#' @param X data matrix 
#' @param ken kernel type 
#' @param p dimension of random feature space 
#' @param H.W random weights as computed by \code{\link[ELR]{Weights.ELR}}
#' @param n sample size 
#' @param d dimension 
#' @param n.class number of classifications tasks
#' @param n.reg number of regression tasks    
#' @param class.prior proportion of positive examples 
#' @param ... Further arguments passed to or from other methods. 
#' @return Randomize kernel matrix from \code{Kernel.HW } and list of 
#  data matrices from \code{SynData.MultiTaskELR} and corresponding task type 
#' @references
#' Che Ngufor, Sudhindra Upadhyaya, Dennis Murphree, Nageswar R. Madde, 
#' Daryl J. Kor, and Jyotishman Pathak. "A Heterogeneous Multi-task 
#' Learning for Predicting RBC Transfusion and Perioperative Outcomes". 
#' To appear in the  15th Conference on Artificial Intelligence in Medicine, 
#' AIME 2015, Pavia, Italy, June 17-20, 2015. Proceedings. 
#'
#' @author  Che Ngufor <Ngufor.Che@@mayo.edu>
#'
#' @import ELR mvtnorm
NULL
#' @rdname Utils 	
Kernel.HW <- function(form, X, ken, p, H.W = NULL) {
rhs.vars <- rhs.form(form)
X1 = data.matrix(X[, rhs.vars]) 
if(is.null(H.W)) {
resp <- lhs.form(form)
Y <- matrix(X[,resp])
H.W <- Weights.ELR(X1, p, ken)
Q <-Kernel.ELR(X1, p, H.W, ken)
res <- list(Q=Q, W = H.W, Y=Y) 
}else {
res <- Kernel.ELR(X1, p, H.W, ken)
}
return(res)
}
#' @rdname Utils
#' @export
SynData.MultiTaskELR <- function(n=100, d=10, n.class = 3, n.reg = 2,
         class.prior = 0.5){
ix = seq(0, 1.5, length.out  = 2) 
u = t(sapply(1:2, function(x) rep(ix[x], d))) 
sigma =  diag(x = 0.5, nrow = d, ncol = d)
pr <- dbinom(c(1,2), size=2, prob=class.prior)
dat.c <- lapply(1:n.class, function(y) {
dat <- do.call(rbind.data.frame, lapply(c(-1, 1), function(x) 
cbind.data.frame(key = rep((x+1)/2, ifelse(floor(n*pr[x]) >= 10, floor(n*pr[x]), 10)), 
rmvnorm(ifelse(floor(n*pr[x]) >= 10, floor(n*pr[x]), 10), mean = u[x,], sigma = sigma))
)
)
names(dat) <- c(paste0("class", y), paste0("V", 1:d))
dat[sample(nrow(dat), nrow(dat)), ]
})

dat.r <- lapply(1:n.reg, function(x) {
Y <- apply(2*sin(dat.c[[x]][,-1]) - cos(pi*dat.c[[x]][,-1]) + 2, 1, min) 
dat.r <- cbind(Y, dat.c[[x]][,-1])
colnames(dat.r) <-c(paste0("reg", x), paste0("V", 1:d))
return(dat.r)
})

TT <- c(rep("class", n.class), rep("regression", n.reg))
dat <- append(dat.c,dat.r)
res <- list(dat = dat, task.type = TT)
return(res)
}












