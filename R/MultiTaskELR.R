#' MultiTaskELR  
#' 
#' This function trains a regularization based multi-task learning algorithm 
#' using extreme logistic regression \code{\link[ELR]{ELR}} for 
#' classification tasks and extreme learning machine (ELM) for regression 
#' tasks.  
#' @name MultiTaskELR
#' @param form A named list of formulas for each task. 
#' The response variable of each formula corresponds to 
#' one of the elements in the vector 'resp.vars' which 
#' equally is the name of the corresponding list element i.e 
#' names(form) = resp.vars. 
#' @param dat a named list (names(dat) = resp.vars) of matrices  
#' for each task all of the same dimension and type of predictors. 
#' The response variables  are all  binary 0,1 numeric vector.
#' @param resp.vars  a character vector of response variables for each task.  
#' @param task.type  a named character vector (names(task.type) = resp.vars) 
#' of type of tasks (class/regression)  
#' @param para A named parameter list with required  entries:
#' \itemize{
#' \item{p: }{Dimension of random feature space }
#' \item{gamma: }{Numeric regularization parameter }
#' \item{mu: }{Task similarity parameter }
#' \item{ken: }{Type of kernel function}
#' }
#' @param \dots Further arguments passed to or from other methods.
#' @return An object of class \code{MultiTaskELR}; a list with items 
#' \item{form}{list of formulas for each task}
#' \item{beta}{estimated parameters for all tasks}
#' \item{H.W}{Random weight matrix} 
#' \item{resp.vars}{a character vector of response variables for each task}
#' \item{task.type}{a named character vector of type of tasks}
#' \item{para}{A named parameter list}
#'
#' @details \code{MultiTaskELR} trains a multi-task learning mpodel using either 
#' \code{\link[ELR]{ELR}} for classification tasks or extreme learning machine 
#'  ELM for regression tasks. 
#' The algorithm is based on the the classical regularization based MTL 
#' algorithm. Currently, it is assumed that all 
#' tasks have the same number of observations and type of predictors. 
#' If only one task is specified, then this is equivalent to 
#' \code{\link{ELR}} or ELM.  
#'
#' @seealso  \code{\link[ELR]{ELR}} for single task
#' learning   
#'
#' @references
#' [1] Che Ngufor, Sudhindra Upadhyaya, Dennis Murphree, Nageswar R. Madde, 
#' Daryl J. Kor, and Jyotishman Pathak. "A Heterogeneous Multi-task 
#' Learning for Predicting RBC Transfusion and Perioperative Outcomes". 
#' To appear in the  15th Conference on Artificial Intelligence in Medicine, 
#' AIME 2015, Pavia, Italy, June 17-20, 2015. Proceedings. 
#'
#' [2] Theodoros Evgeniou, and Massimiliano Pontil. "Regularized multi--task 
#' learning." Proceedings of the tenth ACM SIGKDD international conference on 
#' Knowledge discovery and data mining. ACM, 2004.
#' 
#' @author  Che Ngufor <Ngufor.Che@@mayo.edu>
#' @import matrixcalc MASS ELR Matrix
NULL 
#'
#' @rdname MultiTaskELR 
#' @export
MultiTaskELR  <- function(form, ...) UseMethod("MultiTaskELR")
#' @rdname MultiTaskELR
#' @export
#' @examples
#' \dontrun{
#' set.seed(12345)
#' dat <- SynData.MultiTaskELR()
#' task.type = dat$task.type 
#' ix = sample(nrow(dat$dat[[1]]), floor(nrow(dat$dat[[1]])*0.75))
#'  
#' dd.trn <- lapply(dat$dat, function(y) y[ix, ])
#' dd.tst <- lapply(dat$dat, function(y) y[-ix, ])
#' 
#' resp.vars <- sapply(dd.trn, function(x) colnames(x)[1])
#'  rhs.vars <- names(dat$dat[[1]])[-1]
#'  names(dd.trn) = names(dd.tst) = resp.vars
#'  names(task.type) = resp.vars
#'  
#'  form <- lapply(resp.vars, function(x) as.formula(paste0(paste0(x, "~"), 
#'               paste0(rhs.vars, collapse= "+"))))              
#'  names(form) = resp.vars
#'  
#'  para <- list( ken = "sigmoid", p = 100, gamma = 10.01, mu = 0.05)
#'  
#'  mtl.mod <- MultiTaskELR(form, dd.trn, resp.vars, task.type, para)
#'  
#'  pred <- predict(mtl.mod, dd.tst, class.type = "class")
#'  
#'  lapply(resp.vars[task.type == "class"], function(x) 
#'            table(true = dd.tst[[x]][, x], pred = pred$class.pred[,x]))
#' 
#' require(DMwR)
#' 
#'  lapply(resp.vars[task.type == "regression"], function(x) 
#'           regr.eval(dd.tst[[x]][, x], pred$reg.pred[,x]))
#' }
#'
MultiTaskELR.default <- function(form, dat, resp.vars, task.type,  
                        para, ...){
ken = para$ken
gamma = para$gamma
p=para$p
mu=para$mu

sig <- ifelse(task.type == "class", 1, 0)
Q.W <- lapply(1:length(resp.vars), function(x) Kernel.HW(form[[x]], dat[[x]], 
             ken, p, H.W = NULL))
H.W <- lapply(Q.W, function(x) x$W)
names(H.W) <- resp.vars

Q <- lapply(Q.W, function(x) x$Q) 
Y <- do.call(rbind, lapply(Q.W, function(x) x$Y)) 

n = nrow(Q[[1]])
nT = length(task.type)

I.T <- do.call(rbind, lapply(Q, function(x) x/sqrt(mu)))
Q.T <- cbind(I.T,  as.matrix(bdiag(Q)))

########################################

A <- t(Q.T)%*%Q.T  
in.gamma <- (4^sig)/gamma
Ip  = matrix(rep(1, p*(nT+1)))
Ip[-c(1:p)]  <- do.call(c, lapply(2:(length(sig)+1), 
               function(x) in.gamma[x-1]*Ip[((x-1)*p+1):(x*p)]))
diag(A) <- diag(A) + Ip 
diag(A) <- diag(A) + 1e-12

z.q <- t(Q.T)%*%do.call(c,lapply(1:length(task.type), function(x) 
4^sig[x]*(Y[((x-1)*n+1):(x*n)]-0.5*sig[x])))
beta <- solve(A, z.q)
res =  list(form=form, beta=beta, H.W = H.W, resp.vars = resp.vars, 
             task.type=task.type, para=para)
class(res) = "MultiTaskELR"
return(res)
}














