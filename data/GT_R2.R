library(jsonlite)
library(ggplot2)
require(gridExtra)

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}

setwd("~/GitHub/LearningToInfer/data")

fn = "N_part19__expt_nameGTstudy1__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch300__train_lr0.01__plot_data"
data <- fromJSON(txt=fn)

true_llo = log(log_odds(data$hrms))
model_llo = log(log_odds(data$ams))
strength_l = log(abs(data$strength))
weight_l = log(data$weight*33)
exact = rep(c(rep(TRUE, 300), rep(FALSE, 300)), 20)

# Random sample from generative 
df = data.frame(strength_l = strength_l[!exact],
                weight_l = weight_l[!exact],
                y = true_llo[!exact])


corr <-lm(df$y - log(log(0.6/0.4)) ~ df$strength_l)
summary(corr)

p1<- ggplot(df,aes(x = strength_l,y =y), alpha = 0.001)+#stat_summary(fun.data=mean_cl_normal) + 
  geom_jitter(size = 0.001, width = 0.18, height = 0.1)+
  ylim(c(-1.5, 2.5))+
  xlim(c(-4, 0.5))+
  geom_smooth(method='lm') + annotate("text", x = -2, y = 2.2, 
                                      label = paste("R2 = ", as.character(round(summary(corr)$r.squared, digits = 3))))
p1

corr <-lm(df$y - log(log(0.6/0.4)) ~ df$weight_l)
summary(corr)

p2<- ggplot(df,aes(x = weight_l,y =y), alpha = 0.001)+#stat_summary(fun.data=mean_cl_normal) + 
  geom_jitter(size = 0.001, width = 0.18, height = 0.1)+
  ylim(c(-1.5, 2.5))+
  xlim(c(0.5, 4))+
  geom_smooth(method='lm') + annotate("text", x = 1.7, y = 2.2, 
                                      label = paste("R2 = ", as.character(round(summary(corr)$r.squared, digits = 3))))
p2

grid.arrange(p1, p2, nrow = 1)
g <- arrangeGrob(p1, p2, nrow = 1)
ggsave(file = "GT_rsquare_sample.png", g)

# Exact Stimuli

df = data.frame(strength_l = strength_l[exact],
                weight_l = weight_l[exact],
                y = true_llo[exact])


corr <-lm(df$y - log(log(0.6/0.4)) ~ df$strength_l)
summary(corr)

p1<- ggplot(df,aes(x = strength_l,y =y), alpha = 0.001)+#stat_summary(fun.data=mean_cl_normal) + 
  geom_jitter(size = 0.001, width = 0.18, height = 0.1)+
  ylim(c(-1.5, 2.5))+
  xlim(c(-4, 0.5))+
  geom_smooth(method='lm') + annotate("text", x = -2, y = 2.2, 
                                      label = paste("R2 = ", as.character(round(summary(corr)$r.squared, digits = 3))))
p1

corr <-lm(df$y - log(log(0.6/0.4)) ~ df$weight_l)
summary(corr)

p2<- ggplot(df,aes(x = weight_l,y =y), alpha = 0.001)+#stat_summary(fun.data=mean_cl_normal) + 
  geom_jitter(size = 0.001, width = 0.18, height = 0.1)+
  ylim(c(-1.5, 2.5))+
  xlim(c(0.5, 4))+
  geom_smooth(method='lm') + annotate("text", x = 1.7, y = 2.2, 
                                      label = paste("R2 = ", as.character(round(summary(corr)$r.squared, digits = 3))))
p2

grid.arrange(p1, p2, nrow = 1)
g <- arrangeGrob(p1, p2, nrow = 1)
ggsave(file = "GT_rsquare_stimuli.png", g)

#**************************************
