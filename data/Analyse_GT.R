library(jsonlite)
library(ggplot2)
require(gridExtra)


setwd("~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}
cbbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

fn = "N_part19__alpha0.01__expt_nameGTstudy1__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch300__train_lr0.01__plot_data"
data <- fromJSON(txt=fn)

true_llo = log(log_odds(data$hrms))
model_llo = log(log_odds(data$ams))
strength_l = log(abs(data$strength))
weight_l = log(data$weight*33)
exact = rep(c(rep(TRUE, 300), rep(FALSE, 0)),20)

model_lm0 <- lm(true_llo - log(log(0.6/0.4)) ~ strength_l + weight_l + 0)
summary(model_lm0)

model_lm_random <- lm(model_llo[!exact] - log(log(0.6/0.4)) ~ strength_l[!exact] + weight_l[!exact] + 0)
summary(model_lm_random)

model_lm_stimuli <- lm(model_llo[exact] - log(log(0.6/0.4)) ~ strength_l[exact] + weight_l[exact] + 0)
summary(model_lm_stimuli)

df = data.frame(mu = summary(model_lm_stimuli)$coefficients[1:2],
  se = summary(model_lm_stimuli)$coefficients[2:3],
  which = factor(c('Strength', 'Weight'))
)

df$participants <- c(0.88, 0.31)

part <- aes(x = which, y = df$participants)


p1 <- ggplot(df, aes(y=mu, x=which)) + 
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
  theme_classic() +xlab("Predictor")+ylab("Regression coefficient")+
  scale_y_continuous(limits = c(0,1.0), expand = c(0, 0)) +
  #adjust text size
  theme(text = element_text(size=18, family="serif"))+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position = "bottom",
        panel.border = element_rect(colour = "black", fill=NA, size=2))


p1
ggsave(file = "GT_fulltrain.png", p1)

