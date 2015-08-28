# robustMultiTasklssvm 

This package implements the (weighted) LS-SVM and a regularization based multi-task learning with (weighted) LS-SVM (MTL WLS-SVM) algorithms. Instead of using the standard SVM kernel matrix, the randomized feature based kernel  approach in Extreme Logistic Regression  ([ELR](http://link.springer.com/article/10.1007%2Fs11634-014-0194-2)) is implemented. Several robust weight functions such as *Huber, Hample, logistic* and "Myriad* ([De Brabanter et. al](http://link.springer.com/chapter/10.1007%2F978-3-642-04274-4_11) ) are implemented. 

For MTL WLS-SVM, the tasks can be classification or regression. If only  one task is specified, then this is simply WLS-VM as implented by the function *lssvm*.   

Full details of the algorithm will (hopefully) appear in [1], a paper submitted to ICDM 2015. More information on the algorithm and parameters will be provided once the review process is over. 

## Why Use  robustMultiTasklssvm  
  - a simple and very fast LS-SVM algorithm compared to the implementation in the [kernlab](http://cran.r-project.org/web/packages/kernlab/index.html) package  
  - probably the first implementation o a MTL algorithm in R 
 

## How to get Started? 
Install via devtools: 

```sh
> devtools::install_github("nguforche/robustMultiTasklssvm")
```
## Parameters

The performance of WLS-SVM and MTL WLS-SVM in terms of training time and accuracy  crucially depends on the choice of robust weight function and other parameters as in the ELR algorithm: *p* the dimension of the randomized feature space and *gamma* the regularization constant. The *Logistic*, *Hampel* and *Huber* robust weight functions have been found to produce acceptable results for good choices of *p* and *gamma*. The function *lssvm_gridsearch* implements a grid search for optimal *p* and *gamma* for the WLS-SVM algorithm. Similar implementation can be made for MTl WLS-SVM.  

## Examples
#### WLS-SVM  
```sh
 library(robustMultiTasklssvm )
 set.seed(12345)
 dat <- SynData()
 dat$key <- ifelse(dat$key ==1, 1, -1) 
 ix = sample(nrow(dat), floor(nrow(dat)*0.75))
 dat.trn = dat[ix, ]
 dat.tst = dat[-ix, ]
 form <- as.formula(paste("key ~ ", paste(names(dat)[!names(dat)%in%"key"], collapse = "+")))
 para <- list( ken = "sigmoid", p = 100, gamma = 10.01, tol = 1e-6, max.iter = 100, robust = TRUE)
 mod <- lssvm(form, dat.trn, para)
 pred <- predict(mod, dat.tst)
 perf <- Performance(pred$prob[,2], dat.tst$key)
 
```
#### MTLWLS-SVM 
```sh
 set.seed(12345)
 dat <- SynData.lssvm()
 task.type = dat$task.type 
 ix = sample(nrow(dat$dat[[1]]), floor(nrow(dat$dat[[1]])*0.75))
 dd.trn <- lapply(dat$dat, function(y) y[ix, ])
 dd.tst <- lapply(dat$dat, function(y) y[-ix, ])  resp.vars <- sapply(dd.trn, function(x) colnames(x)[1])
 rhs.vars <- names(dat$dat[[1]])[-1]
 names(dd.trn) = names(dat.tst) = resp.vars
 names(task.type) = resp.vars
 form <- lapply(resp.vars, function(x) as.formula(paste0(paste0(x, "~"),  
       paste0(rhs.vars, collapse= "+")))) 
 names(form) = resp.vars
 para <- list( ken = "sigmoid", p = 100, gamma = 10.01, mu = 0.05, tol = 1e-6, 
            max.iter = 200,weight.fun = list(fun="Logistic"), robust = TRUE )
 
 mtl.mod <- robustMultiTasklssvm(form, resp.vars, dd.trn, task.type,  para)
 pred <- predict(mtl.mod, dd.tst, class.type = "class")
 lapply(resp.vars[task.type == "class"], function(x) 
           table(true = dd.tst[[x]][, x], pred = pred$class.pred[,x]))
 lapply(resp.vars[task.type == "regression"], function(x) 
          regr.eval(dd.tst[[x]][, x], pred$reg.pred[,x]))

```
## Limitations

Currently LS-SVM and MTL LS-SVM are implemented only for binary classification tasks. Training of the MTL algorithm can be slow for large datasets. The primary reason for this is that he kernel matrix is impleneted as a full matrix and the sparseness is not taken advantage of. 

## Todo's
 - implement sparse approximations and solvers 
 - tasks clustering.
 
## License
* GPL(>= 3)

## References
  [1] Che Ngufor, Dennis Murphree, Sudhindra Upadhyaya, Daryl J. Kor, 
    and Jyotishman Pathak. "Robust Multi-task Learning Using Weighted LS-SVM 
   for Predicting Re-operation due to Bleeding". Submitted to ICDM 2015, 
  November 14-17, 2015, Atlantic City, NJ USA



