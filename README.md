# ELR Version 0.1 

Extreme Logistic Regression [ELR](http://link.springer.com/article/10.1007%2Fs11634-014-0194-2) is an extension of the Extreme Learning Machine [ELM](http://www.ncbi.nlm.nih.gov/pubmed/21984515) algorithm to kernel logistic regression (KLR). Briefly, the input data is mapped onto a randomized feature space whose dimension can be chosen by the user and the resulting non-linear system can be solved with any existing solver.Two algorithms are implemented in the package: iterative reweighted least-squares ELR (IRLS-ELR) and least-squares ELR (LS-ELR), a simple approximation of the logistic functiom which converts the non-linear system of IRLS-ELR to a linear system. Because the dimension of the randomized feature space can be chosen, the algorithm can be extremely fast to train and scalable to very large data sets.     

## Why Use ELR
  - simple and efficient 
  - fast training and testing
  - outputs class probabilities 
  - scalable 
  - very compertative with many state of the arts methods like SVM, KLR, random forest, ELM, neural network, etc.  

## How to get Started? 
Install via devtools: 

```sh
> devtools::install_github("nguforche/ELR")
```
## Parameters

Just like many machine learning algorithms (such as SVM and KLR), generalization performance crucially depends on the choice of turning paramters. However, a major advantage of ELR is that only the regularization parameter *gamma* need to be tuned. Good generalization performance can be obtained by setting the dimension of the randomized feature space *p* large enough (such as *p* >= 1000). Regardless, optimal choices for *gamma* and *p* can be selected by cross-validation. The computational cost of this exercise is often significantly lessen because of the simplicity of the method.   



## Examples
#### Artificial data 
```sh
> library(ELR)
> set.seed(12345)
> dat <- SynData() # generate artificial data 
> ix = sample(nrow(dat), floor(nrow(dat)*0.75))
> dat.trn = dat[ix, ]
> dat.tst = dat[-ix, ]
> form <- as.formula(paste("key ~ ", paste(names(dat)[!names(dat)%in%"key"], collapse = "+")))
> para <- list( ken = "sigmoid", p = 100, gamma = 10.01)
> mod <- ELR(form, dat.trn, para, model="LS-ELR")
> pred <- predict(mod, dat.tst)
> perf <- Performance(pred$prob[,2], dat.tst$key)

```
#### Iris Data 
```sh
> library(ELR)
> set.seed(12345)
> dat$Species <- as.factor(ifelse(dat$Species == "versicolor", 1, 0))
> plot(dat$Petal.Length, dat$Petal.Width, pch=c(8, 10)[unclass(dat$Species)], 
    col=c("red", "blue")[unclass(dat$Species)], main="", xlab = "Petal length", 
    ylab = "Petal Width")
> ix = sample(nrow(dat), floor(nrow(dat)*0.5))
> dat.trn = dat[ix, ]
> dat.tst = dat[-ix, ]
> form <- as.formula(paste("Species ~ ", paste(names(dat)[!names(dat)%in%"Species"], collapse = "+")))
> para <- list( ken = "sigmoid", p = 300, gamma = 10.01, tol = 1e-6, max.iter = 100)
> elr.mod <- ELR(form, dat.trn, para, model="LS-ELR")
> pred.elr <- predict(elr.mod, dat.tst)
> perf.elr <- Performance(pred$prob[,2], dat.tst$Species)
> # logistic regression 
> glm.mod <- glm(form,data=dat.trn,family=binomial())
> pred.glm <- predict(glm.mod, dat.tst, type = "response")
> perf.glm <- Performance(pred.glm, dat.tst$Species)
> # random forest 
> require(randomForest)
> mod.rf <- randomForest(form,  data = dat.trn, ntree = 2000) 	
> pred.rf <-  predict(mod.rf, newdata = dat.tst, type = "prob") 
> perf.rf <- Performance(pred.rf[,2], dat.tst$Species)
```


## Limitations/Bugs 

Currently only for binary classification tasks. However, it should not be too hard to extend to multi-class. There is a variational Bayes implementation of the algorithm for multi-class, I just haven't got arround to code it up into a package. Send me an email if your are interested in working on it. 

As this is the first version of the package, I wont be supprise if you find tons of bugs :). Please let me know here or through email if you find any issues. 


### Todo's
 - Write package for multi-class
 - Add more examples and code comments
 
## License
* GPL(>= 3)

## References
 * Ngufor, Che, and Janusz Wojtusiak. "Extreme logistic regression." 
 Advances in Data Analysis and Classification (2015): 1-26.


[1]: http://example.com/ "Title"




