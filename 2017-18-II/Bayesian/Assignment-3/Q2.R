library(truncnorm)
data <- read.csv("C:/Users/Tanav Jain/Downloads/binary.csv")
admit<-data[,"admit"]
gre<-data[,"gre"]
ln_gre<-log(gre)
gpa<-data[,"gpa"]
rank<-data[,"rank"]
Rank<-factor(rank)
myprobit <- glm(admit ~ ln_gre + gpa + Rank, family = binomial(link = "probit"),data)
y<-admit
rank2<- rep(0, length(y))
rank3<- rep(0, length(y))
rank4<- rep(0, length(y))
for (i in 1:length(y))
{
if (rank[i]==2){
rank2[i]=1}
else if (rank[i]==3){
rank3[i]=1}
else if (rank[i]==4){
rank4[i]=1}
}
const <- rep(1,length(y))
X<-cbind(const, ln_gre, gpa, rank2, rank3, rank4 )


k <- 6
n <- 12501

betap <- matrix (0L,k,n)
betap[,1]<- rep(1,k)
xtx <- crossprod(X)
xty <- crossprod(X,y)

beta0 <- rep(0,k) 
B0 <- 100*diag(k)
B0inv <- ginv(B0)
B0b0 <- B0inv%*%beta0

B1<- ginv((xtx) + B0inv)

for (i in 2:n) {

zg<- rep(0,length(y))
    for (j in 1:length(y))
{
	xb<- X[j,]%*%betap[,i-1]
	if (y[j]==0){
	zg[j]<- rtruncnorm(n=1, a=-Inf, b=0, mean=xb, sd=1)
}
	else{
	zg[j]<- rtruncnorm(n=1, a=0, b=Inf, mean=xb, sd=1)
}
}
    xtz<-crossprod(X,zg)
    betabar <- B1%*%(xtz + B0b0)
    betap[,i] <- mvrnorm(n=1,betabar, B1)
    
}
Beta <- betap[,2502:12501]
Mean<- rep(0,k)
Std<- rep(0,k)
for (i in 1:k){
Mean[i] <- mean(Beta[i,])
Std[i] <- sd(Beta[i,])
}
Mean
Std
summary(myprobit)