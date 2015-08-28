# mtlELR 

This is perhaps the first R package for multi-task learning. The package implements the regularized mult-task learning algorithm of [1] using the  Extreme Logistic Regression  ([ELR](http://link.springer.com/article/10.1007%2Fs11634-014-0194-2)).  

The learning tasks can be classification or regression. If only  one task is specified, then this is simply ELR as implented by the function *ELR* in the package https://github.com/nguforche/ELR.   

Full details of the algorithm will appear in [2], a paper accepted in the 
[IEEE DSAA'2015](http://dsaa2015.lip6.fr/). 


## Why Use  mtlELR 
  - a simple regularized MTL algorithm that can improve prediction performaance over single task learning  
  - probably the first implementation of a MTL algorithm in R 
 

## How to get Started? 
Install via devtools: 

```sh
> devtools::install_github("nguforche/mtlELR")
```
## Parameters
The performance of mtlELR in terms of training time and accuracy depends on the choice of *mu* the task similarity parameter, the penalty term *gamma* and *p* the dimension of the randomized feature space of ELR. 

## Examples
####   
```sh
 library(mtlELR )
  set.seed(12345)
 dat <- SynData.MultiTaskELR()
 task.type = dat$task.type 
 ix = sample(nrow(dat$dat[[1]]), floor(nrow(dat$dat[[1]])*0.75))
 dd.trn <- lapply(dat$dat, function(y) y[ix, ])
 dd.tst <- lapply(dat$dat, function(y) y[-ix, ])
 resp.vars <- sapply(dd.trn, function(x) colnames(x)[1])
 rhs.vars <- names(dat$dat[[1]])[-1]
 names(dd.trn) = names(dd.tst) = resp.vars
 names(task.type) = resp.vars
 form <- lapply(resp.vars, function(x) as.formula(paste0(paste0(x, "~"), 
            paste0(rhs.vars, collapse= "+"))))              
 names(form) = resp.vars
 para <- list( ken = "sigmoid", p = 100, gamma = 10.01, mu = 0.05)
 mtl.mod <- MultiTaskELR(form, dd.trn, resp.vars, task.type, para)
 pred <- predict(mtl.mod, dd.tst, class.type = "class")
lapply(resp.vars[task.type == "class"], function(x) 
        table(true = dd.tst[[x]][, x], pred = pred$class.pred[,x]))
 require(DMwR)
 lapply(resp.vars[task.type == "regression"], function(x) 
           regr.eval(dd.tst[[x]][, x], pred$reg.pred[,x]))
 ```
## Limitations

Currently mtlELR is implemented only for binary classification tasks. Training of the algorithm can be slow for large datasets. The primary reason for this is that the randomized kernel matrix is implemented as a full matrix and sparseness is not taken advantage of. 

## Todo's
 - implement sparse approximations and solvers 

## License
* GPL(>= 3)

## References
[1] Evgeniou, Theodoros, and Massimiliano Pontil. "Regularized multi--task learning." Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2004.

[2] Che Ngufor, Sudhindra Upadhyaya, Dennis Murphree, Daryl Kor, and Jyotishman Pathak. "Multi-task Learning with Selective Cross-Task Transfer for Predicting Bleeding and other  Important Patient Outcomes" 
IEEE DSAA'2015 Oct 19-21 Paris, France 



