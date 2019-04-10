library(jsonlite)
library(ggplot2)
require(gridExtra)

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}

setwd("~/GitHub/LearningToInfer/data")

get_df <- function(fn, exact){
  data <- fromJSON(txt=fn)
  
  true_llo = log(log_odds(data$hrms))
  model_llo = log(log_odds(data$ams))
  strength_l = log(abs(data$strength))
  weight_l = log(data$weight*33)
  exact = exact
  
  model_lm0 <- lm(true_llo - log(log(0.6/0.4)) ~ strength_l + weight_l + 0)
  summary(model_lm0)
  # model_lm1 <- lm(model_llo - log(log(0.6/0.4)) ~ strength_l + weight_l + 0)
  # summary(model_lm1)
  # 
  # if sum(exact) < length(exact){
  #   model_lm_exact <- lm(model_llo[!exact] - log(log(0.6/0.4)) ~ strength_l[!exact] + weight_l[!exact] + 0)
  # }
  # else {
  #   model_lm_exact <- model_lm_overall
  # }
  # summary(model_lm_exact)
  # 
  model_lm_overall <- lm(model_llo[exact] - log(log(0.6/0.4)) ~ strength_l[exact] + weight_l[exact] + 0)
  summary(model_lm_overall)
  
  df = data.frame(mu = c(#summary(model_lm_exact)$coefficients[1:2], 
                         summary(model_lm_overall)$coefficients[1:2]),
                  se = c(#summary(model_lm_exact)$coefficients[2:3], 
                         summary(model_lm_overall)$coefficients[2:3]),
                  cond = factor(c(#'Exact', 'Exact', 
                                  'Overall', 'Overall')),
                  which = factor(c(#'Strength', 'Weight', 
                                   'Strength', 'Weight'))
  )
  
  return(df)
}

plot<- function(df, title){
  
  df$participants <- c(0.88, 0.31)
  cbbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
  part <- aes(x = which, y = df$participants)
  
  
  p1 <- ggplot(df, aes(y=mu, x=which, fill=cond)) + 
    #bars
    geom_bar(position="dodge", stat="identity")+
    geom_point(mapping = aes(x = which, y = participants), position = 'dodge')+
    #0 to 1
    #golden ratio error bars
    # geom_errorbar(limits, position="dodge", width=0.31)+
    # #point size
    # geom_point(size=3)+
    scale_fill_manual(values=cbbPalette[c(6,7)])+
    #title
    theme_classic() +xlab("Condition")+ylab("Mean Estimates")+
    ggtitle(title)+
    scale_y_continuous(limits = c(0,1), expand = c(0, 0)) +
    #adjust text size
    theme(text = element_text(size=18, family="serif"))+
    theme(panel.background = element_blank(),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          axis.line = element_line(colour = "black"),
          legend.position = "bottom",
          panel.border = element_rect(colour = "black", fill=NA, size=2))
  
  return(p1) 
}

# With training on exact stimuli
exact = rep(c(rep(TRUE, 300)), 20)
fn1 = "N_part19__expt_nameGTstudy1_exact__NHID5__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch300__train_lr0.01__plot_data"
fn2 = "N_part19__expt_nameGTstudy1_exact__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch300__train_lr0.01__plot_data"

df = get_df(fn1, exact)
p1 = plot(df, "High expressivity")

df = get_df(fn2, exact)
p2 = plot(df, "Low expressivity")

grid.arrange(p1, p2, nrow = 1)
g <- arrangeGrob(p1, p2, nrow = 1)
ggsave(file = "GT_exacttrain.png", g)

# # With training on full generative
# exact = rep(c(rep(TRUE, 300), rep(FALSE, 300)), 20)
exact = rep(c(rep(TRUE, 300)), 20)
fn1 = "N_part4__expt_nameGTstudy1__NHID1__NONLINrbf__L20.0__test_epoch1__test_lr0.001__train_epoch300__train_lr0.01__plot_data"

df = get_df(fn1, exact)
p1 = plot(df, "Model predictions")
p1
ggsave(file = "GT_fulltrain.png", p1)

#************************************
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
