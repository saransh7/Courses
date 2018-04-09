library(MASS)
library("xlsx")
data <-
  read.csv("/home/saransh/Desktop/Assignment3_bayesian/hsbdemo.csv")
read<-data[,"read"]
read1<-read/10
write<-data[,"write"]
write1<-write/10
math<-data[,"math"]
math1<-math/10
science<-data[,"science"]
science1<-science/10
socst<-data[,"socst"]
socst1<-socst/10
female<-data[,"female"]
academic<-data[,"academic"]
general<-data[,"general"]
private<-data[,"private"]
middle<-data[,"middle"]
high<-data[,"high"]
C1<-lm(read1~1+female+middle+high+private+academic+general+socst1,data)
C2<-lm(math1~1+female+middle+high+private+academic+general+science1,data)

y1<-read1
y2<-math1
const <- rep(1,200)
X1<-cbind(const, female, middle, high, private, academic, general, socst1)
X2<-cbind(const, female, middle, high, private, academic, general, science1)


k <- 8
n <- 12500

betap <- matrix (0L,k,n)
sigma2 <- rep (0, n+1)
sigma2[1] <- 20
xtx <- crossprod(X1)
xty <- crossprod(X1,y1)


beta0 <- rep(0,k) 
B0 <- 10*diag(k)
B0inv <- ginv(B0)
B0b0 <- B0inv%*%beta0

alpha0 <- 5
alpha1 <- 205
del <- 8

for (i in 1:n) {
  B1 <- ginv(((1/sigma2[i])*xtx) + B0inv )
  betabar <- B1%*%((1/sigma2[i])*xty + B0b0)
  betap[,i] <- mvrnorm(n=1,betabar, B1)
  
  delg = del + crossprod(( y1-X1%*%betap[,i]))
  
  temp <- rgamma(n=1, shape=alpha1/2, rate=delg/2)
  sigma2[i+1] = 1/temp
}
Beta <- betap[,2501:12500]
Mean<- rep(0,k)
Std<- rep(0,k)
for (i in 1:k){
  Mean[i] <- mean(Beta[i,])
  Std[i] <- sd(Beta[i,])
}
summary(C1)
Mean
Std
mean(sigma2[2502:12501])
sd(sigma2[2502:12501])

k <- 8
n <- 12500

betap <- matrix (0L,k,n)
sigma2 <- rep (0, n+1)
sigma2[1] <- 20
xtx <- crossprod(X2)
xty <- crossprod(X2,y2)


beta0 <- rep(0,k) 
B0 <- 10*diag(k)
B0inv <- ginv(B0)
B0b0 <- B0inv%*%beta0

alpha0 <- 5
alpha1 <- 205
del <- 8

for (i in 1:n) {
  B1 <- ginv(((1/sigma2[i])*xtx) + B0inv )
  betabar <- B1%*%((1/sigma2[i])*xty + B0b0)
  betap[,i] <- mvrnorm(n=1,betabar, B1)
  
  delg = del + crossprod(( y2-X2%*%betap[,i]))
  
  temp <- rgamma(n=1, shape=alpha1/2, rate=delg/2)
  sigma2[i+1] = 1/temp
}
Beta <- betap[,2501:12500]
Mean<- rep(0,k)
Std<- rep(0,k)
for (i in 1:k){
  Mean[i] <- mean(Beta[i,])
  Std[i] <- sd(Beta[i,])
}
summary(C2)
Mean
Std
mean(sigma2[2502:12501])
sd(sigma2[2502:12501])

X<-matrix(0L,400,16)
y<-matrix(0L,400,1)
check<-0
j<-1
for (i in 1:400){
  if(check==0){
    X[i,1:8]<- X1[j,]
    y[i]<-y1[j]
    check<-1
  }
  else{
    X[i,9:16]<- X2[j,]
    y[i]<-y2[j]
    check<-0
    j<-j+1
  }
}
k<-16
n<-12500
betap <- matrix (0L,k,n)
E<- matrix(0L,2,2*n+2)
E[,1:2]<- rep(1,4)



beta0 <- rep(0,k) 
B0 <- diag(k)
B0inv <- ginv(B0)
B0b0 <- B0inv%*%beta0

v0<-6
R0<- diag(2)
v1<-206


for (i in 1:n) {
  xex <- rep(0,k) 
  xex
  xey <- rep(0,k) 
  j<-1
  while (j <= 200)
  {
    A<-X[(2*j-1):(2*j),]
    A1<-ginv(E[,(2*i-1):(2*i)])
    A2<-y[(2*j-1):(2*j)]
    xey <- xey + (crossprod(A,A1))%*%A2
    xex <- xex + (crossprod(A,A1))%*%A
  }
  B1 <- ginv(xex + B0inv )
  betabar <- B1%*%(xey + B0b0)
  betap[,i] <- mvrnorm(n=1,betabar, B1)
  sum <- matrix(0L,2,2) 
  
  
  
  for (j in 1:200)
  {
    sum <- sum+ (y[(2*j-1):(2*j)]-X[(2*j-1):(2*j),]%*%betap[,i])%*%t(y[(2*j-1):(2*j)]-X[(2*j-1):(2*j),]%*%betap[,i])
  }
  R1g <- ginv(ginv(R0) + sum)
  
  temp <- rWishart(n=1,v1, R1g)
  E[,(2*i+1):(2*i+2)]<- ginv(temp)
}
Beta <- betap[,2501:12500]
Mean<- rep(0,k)
Std<- rep(0,k)
for (i in 1:k){
  Mean[i] <- mean(Beta[i,])
  Std[i] <- sd(Beta[i,])
}
Mean
Std
mean(sigma2[2502:12501])
sd(sigma2[2502:12501])
