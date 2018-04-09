
library("MASS", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.4")
library("MASS", lib.loc="/usr/lib/R/library")
data <- read.csv("~/R/rstudio-xenial-1.0.153-amd64/Birth-Weight.csv")
bwt<-data[,"bwt"]
bwt1<-bwt/10
gestation<-data[,"gestation"]
gestation1<-gestation/10
weight<-data[,"weight"]
weight1<-weight/10
parity<-data[,"parity"]
age<-data[,"age"]
height<-data[,"height"]
smoke<-data[,"smoke"]
m=c(mean(bwt1),mean(gestation1),mean(parity),mean(age),mean(height),mean(weight),mean(smoke))
std=c(sd(bwt1),sd(gestation1),sd(parity),sd(age),sd(height),sd(weight),sd(smoke))

lmfit<-lm(bwt1~1+gestation1+weight1+parity+age+height+smoke,data)
summary(lmfit)
const <- rep(1,length(y))
X<-cbind(const, gestation1, parity, age, height, weight1, smoke)
y<-bwt1

k <- 7
n <- 12500

betap <- matrix (0L,k,n)
sigma2 <- rep (0, n+1)
sigma2[1] <- 20
xtx <- crossprod(X)
xty <- crossprod(X,y)


beta0 <- rep(0,k) 
B0 <- 100*diag(k)
B0inv <- ginv(B0)
B0b0 <- B0inv%*%beta0

alpha0 <- 8
alpha1 <- 8+length(y)
del <- 8

for (i in 1:n) {
    B1 <- ginv(((1/sigma2[i])*xtx) + B0inv )
    betabar <- B1%*%((1/sigma2[i])*xty + B0b0)
    betap[,i] <- mvrnorm(n=1,betabar, B1)
    
    delg = del + crossprod(( y-X%*%betap[,i]))
    
    temp <- rgamma(n=1, shape=alpha1/2, rate=delg/2)
    sigma2[i+1] = 1/temp
}
Beta <- betap[,2501:12500]
mn<- rep(0,k)
sd<- rep(0,k)

Mean<-(200)
Std<-(200)
for (i in 1:k){
Mean[i] <- mean(Beta[i,])
Std[i] <- sd(Beta[i,])
}
Mean
Std
mean(sigma2[2502:12501])
sd(sigma2[2502:12501])

