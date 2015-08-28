#' predict.MultiTaskELR
#' 
#' Make predictions for MTL with ELR and/or ELM for new data 
#' @param object Trained MultiTaskELR model.
#' @param newdata A a named list of matrices 
#' for each task all of the same dimension and type of predictors. 
#' Names should be exactly the same as the names of the response variables 
#' used for training. 
#' @param class.type type of prediction for classification task: 
#'  "class" for class memberships or "prob" for class probabilities. 
#' @param \dots Further arguments passed to or from other methods.
#' @return predictions: a list with predictions 
#' \item{class.pred}{a matrix whose columns are the predicted class 
#' memberships or probabilities for classification tasks} 
#' \item{reg.pred}{a matrix whose columns are the predictions for  
#' regression tasks}   
#' @references
#' [1] Che Ngufor, Sudhindra Upadhyaya, Dennis Murphree, Nageswar R. Madde, 
#' Daryl J. Kor, and Jyotishman Pathak. "A Heterogeneous Multi-task 
#' Learning for Predicting RBC Transfusion and Perioperative Outcomes". 
#' To appear in the  15th Conference on Artificial Intelligence in Medicine, 
#' AIME 2015, Pavia, Italy, June 17-20, 2015. Proceedings. 
#'
#' @author  Che Ngufor <Ngufor.Che@@mayo.edu>
#' @import ELR
#' @export 
#' 
predict.MultiTaskELR <- function(object, newdata, 
         class.type = "class", ...){
if (!inherits(object, "MultiTaskELR")) stop("Object must be a 
          \"MultiTaskELR \"'")

beta = object$beta
H.W <- object$H.W 
p <- object$para$p
ken <- object$para$ken 
form <- object$form 
task.type <- object$task.type 
resp.vars <- object$resp.vars
mu <- object$para$mu

Q <- lapply(1:length(task.type), function(x) 
             Kernel.HW(form[[x]], newdata[[x]], ken, p, H.W[[x]]))

n = nrow(Q[[1]])
nT = length(task.type)
I.T <- do.call(rbind, lapply(Q, function(x) x/sqrt(mu)))
Q.Tst <- cbind(I.T,  as.matrix(bdiag(Q)))
eta <- Q.Tst%*%beta
### classification 
ix.cls <- which(task.type == "class")
if(length(ix.cls) > 0){
if(class.type == "class"){
pred <- sapply(ix.cls, function(x) ifelse(eta[((x-1)*n+1):(x*n)] >= 0, 1, 0)) 
} else {
ww.cls <- sapply(ix.cls, function(x) eta[((x-1)*n+1):(x*n)] )
pred = sigmoid(ww.cls)
}
colnames(pred) <- resp.vars[ix.cls]
}
### regression
ix.reg <- which(task.type == "regression")
if(length(ix.reg) > 0){
ww.reg <- sapply(ix.reg, function(x) eta[((x-1)*n+1):(x*n)])
colnames(ww.reg) <- resp.vars[ix.reg]
}

return(list(class.pred = pred, reg.pred = ww.reg))
}






