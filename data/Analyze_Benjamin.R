library(jsonlite)
library(ggplot2)
require(gridExtra)
require(plyr)


setwd("~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}

to_theta <- function(x) {
  return(x/(1.0 + x))
}


get_AR <- function(model_los, lik_los){
  summary = summary(lm(model_los ~ lik_los + 0, na.action=na.exclude))$coefficients
  print(summary)
  AR = summary[1,1]
  
  return(AR)
}


do_analysis <- function(fn){
  data0 <- fromJSON(txt=fn)
  data = data.frame(prior_los = log_odds(data0$priors),
                    true_los = log_odds(data0$hrms),
                    model_los = log_odds(data0$ams),
                    Ns = data0$weights*33,
                    thetas = to_theta(data0$lik_ratios)
                    )
  
  data$lik_los = data$true_los - data$prior_los#-log(data0$lik_ratios),
  # data$prior_bins = .bincode(priors, breaks = c(0.0, 0.25, 0.48, 0.52, 0.75,))
  
  
  d_summary = ddply(data,.(priors, conds), summarise, get_AR(model_los, lik_los, prior_los))
  size = nrow(d_summary$..1)
  ds = data.frame(conds = d_summary$conds,
                  priors = d_summary$priors,
                  ARs = d_summary$..1
  )
  ds = unique(ds)
  p0 <- ggplot(ds, aes(x = priors, y = ARs, col = factor(conds))) + 
    geom_point()+
    geom_line()+
    ylim(c(-0.25, 6))
  return(p0)
}

#***************************
fn = "BiasFalse__N_part27__expt_nameBenj__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch30__train_lr0.01__plot_data"
# fn = "BiasFalse__N_part49__expt_nameBenj__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch300__train_lr0.01__plot_data"
data0 <- fromJSON(txt=fn)
data = data.frame(prior_los = log_odds(data0$priors),
                  true_los = log_odds(data0$hrms),
                  model_los = log_odds(data0$ams),
                  Ns = data0$weights*33,
                  sub_exp = data0$sub_exp,
                  thetas = to_theta(data0$lik_ratios)
)

data$lik_los = data$true_los - data$prior_los

data$lik_los[which(!is.finite(data$lik_los))] = NA
data$lik_los[is.nan(data$lik_los)] = NA

data$model_los[which(!is.finite(data$model_los))] = NA
data$model_los[is.nan(data$model_los)] = NA




filter1 = data$model_los > -1500
filter2 = data$model_los < 1500
data$filter = filter1*filter2
p <- ggplot(data, aes(x = true_los*filter, y = model_los*filter)) + 
  geom_point() + xlim(c(-11, 16)) + ylim(c(-4, 4))+ geom_smooth(method = 'loess') + 
  stat_smooth(method = 'lm') +
  geom_abline(intercept = 0.0, slope = 1.0, col = 'red')
p

# Ns
# data$random_reassign = sample(c(1, 2, 3), replace = TRUE, size = length(data$Ns))
d_summary = ddply(data,.(Ns, sub_exp), summarise, get_AR(model_los, lik_los))
ds = data.frame(Ns = d_summary$Ns,
                ARs = d_summary$..1
)
ds = unique(ds)

p <- ggplot(ds, aes(x = Ns, y = ARs)) + 
  geom_point() + xlim(c(-1, 40)) + ylim(c(-1, 2)) + #geom_smooth(method = 'loess') + 
  stat_smooth(method = 'lm') +
  geom_abline(intercept = 1.0, slope = 0.0, col = 'red')
p

# diagnosticity
data$theta_bins = .bincode(data$thetas, breaks = seq(0, max(data$thetas), 0.1))
d_summary = ddply(data,.(theta_bins), summarise, get_AR(model_los, lik_los))
ds = data.frame(thetas = d_summary$theta_bins*0.1,
                ARs = d_summary$..1
)
ds = unique(ds)

p <- ggplot(ds, aes(x = thetas, y = ARs)) + 
  geom_point() + xlim(c(-0.0, 1.01)) + ylim(c(-0.5, 1.5)) + #geom_smooth(method = 'loess') + 
  stat_smooth(method = 'lm') +
  geom_abline(intercept = 1.0, slope = 0.0, col = 'red')
p

# ggsave(file = "PM_control_notrain.png", plot)

