library(jsonlite)
library(ggplot2)
require(gridExtra)

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}

setwd("~/GitHub/LearningToInfer/data")

fn = "N_part1__expt_nameGTstudy1__NHID2__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch200__train_lr0.01__plot_data"
data <- fromJSON(txt=fn)

switch = 2*(log_odds(data$hrms) > 0) - 1
cut = switch*data$strength > -100

true_llo = log(switch[cut]*log_odds(data$hrms)[cut])
model_llo = log(switch[cut]*log_odds(data$ams)[cut])
strength_l = log(switch[cut]*data$strength[cut])
weight_l = log(data$weight*33)[cut]

model_lm0 <- lm(true_llo ~ strength_l + weight_l)
summary(model_lm0)
model_lm1 <- lm(model_llo - model_lm0$coefficients[1] ~ strength_l + weight_l + 0)
summary(model_lm1)
model_lm2 <- lm(model_llo ~ strength_l + weight_l)
summary(model_lm2)
