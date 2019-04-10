library(jsonlite)
library(ggplot2)
require(gridExtra)

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}

setwd("~/GitHub/LearningToInfer/data")

get_df <- function(fn){
  data <- fromJSON(txt=fn)
  
  true_llo = log(log_odds(data$hrms))
  model_llo = log(log_odds(data$ams))
  strength_l = log(abs(data$strength))
  weight_l = log(data$weight*33)
  exact = rep(c(rep(TRUE, 300), rep(FALSE, 300)), 20)
  
  model_lm0 <- lm(true_llo - log(log(0.6/0.4)) ~ strength_l + weight_l + 0)
  summary(model_lm0)
  model_lm1 <- lm(model_llo - log(log(0.6/0.4)) ~ strength_l + weight_l + 0)
  summary(model_lm1)
  
  model_lm_exact <- lm(model_llo[!exact] - log(log(0.6/0.4)) ~ strength_l[!exact] + weight_l[!exact] + 0)
  summary(model_lm_exact)
  
  model_lm_overall <- lm(model_llo[exact] - log(log(0.6/0.4)) ~ strength_l[exact] + weight_l[exact] + 0)
  summary(model_lm_overall)
  
  df = data.frame(mu = c(summary(model_lm_exact)$coefficients[1:2], 
                         summary(model_lm_overall)$coefficients[1:2]),
                  se = c(summary(model_lm_exact)$coefficients[2:3], 
                         summary(model_lm_overall)$coefficients[2:3]),
                  cond = factor(c('Exact', 'Exact', 'Overall', 'Overall')),
                  which = factor(c('Strength', 'Weight', 'Strength', 'Weight'))
  )
  
  return(df)
}

plot<- function(df, title){
  cbbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
  limits <- aes(ymax = mu + se, ymin=mu - se)
  
  
  p1 <- ggplot(df, aes(y=mu, x=which, fill=cond)) + 
    #bars
    geom_bar(position="dodge", stat="identity")+
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

fn = "N_part19__expt_nameGTstudy1__NHID5__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch300__train_lr0.01__plot_data"
df = get_df(fn)
p1 = plot(df, "High expressivity")

fn = "N_part19__expt_nameGTstudy1__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch300__train_lr0.01__plot_data"
df = get_df(fn)
p2 = plot(df, "Low expressivity")

grid.arrange(p1, p2, nrow = 1)
g <- arrangeGrob(p1, p2, nrow = 1)
ggsave(file = "GT.png", g)

# #************************************
# data <- fromJSON(txt=fn)
# 
# true_llo = log(log_odds(data$hrms))
# model_llo = log(log_odds(data$ams))
# strength_l = log(abs(data$strength))
# weight_l = log(data$weight*33)
# exact = rep(c(rep(TRUE, 300), rep(FALSE, 300)), 20)
# 
# df0 = data.frame(strength = strength_l,
#                  weight = weight_l,
#                  which = rep(c(rep('True', 300), rep('Random', 300)), 20))
# 
# p0 <- ggplot(df0, aes(x=strength, col=which)) + 
#   geom_density(bw = 0.5)
# p0
# 
# 
# temp_s = strength_l[exact]
# temp_w = weight_l[exact]
# temp_model = model_llo[exact]
# flag2 = temp_s < median(temp_s)
# 
# model_lm <- lm(temp_model[flag2] - log(log(0.6/0.4)) ~ temp_s[flag2] + temp_w[flag2] + 0)
# summary(model_lm)
# 
# model_lm0 <- lm(temp_model[!flag2] - log(log(0.6/0.4)) ~ temp_s[!flag2] + temp_w[!flag2] + 0)
# summary(model_lm0)
