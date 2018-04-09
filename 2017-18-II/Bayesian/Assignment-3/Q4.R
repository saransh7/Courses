library("MCMCpack")
verbeek <- read.csv("/home/saransh/Desktop/Assignment3_bayesian/Vella-Verbeek 1998/Vella-Verbeek-Data.csv")
y = verbeek["WAGE"]
myvars <- names(verbeek) %in% c("WAGE", "EXPER", "YEAR", "NR") 
X = verbeek[!myvars]
myvars <- names(verbeek) %in% c("EXPER", "")
Intercept = matrix(rep(1,4360), ncol = 1)
colnames(Intercept) <- c("INTERCEPT")
exper = data.matrix(verbeek["EXPER"])
W = cbind(Intercept, exper)


beta_not = matrix(0,32,1)
B0_inv =(1/10)*diag(32)
v0 = 6
D0 = diag(2)
v1 = 6 + 545
alpha_not = 6 
del_not = 3
small_b0 = matrix(0,2,1)
D = diag(2)
beta_panel = matrix(0,32,12500)
beta_panel_bar = matrix(0,32,1)
small_b_panel = array(0 ,dim = c(2,545,12500))
small_b_panel_bar = array(0 ,dim = c(2,545,12500))
alpha_one = alpha_not + 4360
beta_ini = matrix(0,32,1)
small_b0_ini = matrix(1,2,545)
delta_one = matrix(0,1,12500)
hu = matrix(0,12500,1)
B1_panel_i= array(0 , dim = c(8,8,545))
B1 = matrix(0,32,32)
WTW = array(0, dim = c(2,2,545))
D1 = matrix(0,2,2)
D_panel = matrix(0,2,2)
D1_i = array(0, dim = c(2,2,545))

for (i in 1:545) {
  WTW[,,i] = t(W[8*i-7 : 8*i,]) %*% W[8*i-7 : 8*i,]
}

X = as.matrix(X)
Y = as.matrix(Y)

