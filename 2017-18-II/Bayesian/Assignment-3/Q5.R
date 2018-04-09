data <- read.csv("C:/Users/Tanav Jain/Downloads/Q5.csv")
y<-data[,"dep_edu_level"]
mw<-data[,"BG_14_female_work"]
urban<-data[,"BG_14_urban"]
south<-data[,"BG_14_region_south"]
fe<-data[,"BG_edu_father"]
me<-data[,"BG_edu_mother"]
fam<-data[,"BG_fam_income"]
famsq<-sqrt(fam)
f<-data[,"female"]
black<-data[,"race_black"]
age2<-data[,"age_15"]
age3<-data[,"age_16"]
age4<-data[,"age_17"]
const<- rep(1, length(y))
op<-MCMCoprobit(y~1+famsq+me+fe+mw+f+black+urban+south+age2+age3+age4, data=data, burnin=2500, mcmc=10000, b0 = 1, B0 = 100)

summary(op)