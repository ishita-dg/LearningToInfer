library(jsonlite)
library(ggplot2)
require(gridExtra)
require(plyr)


setwd("~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}
to_theta <- function(x) {
  t = x/(1.0 + x)
  return(0.5 + abs(t-0.5))
}
get_AR <- function(model_los, lik_los){
  if (sum(is.na(model_los)) < length(model_los)){
   summary = summary(lm(model_los ~ lik_los + 0, na.action=na.exclude))$coefficients 
   if (length(summary) < 4){
     AR = NA
   }else{
     AR = summary[1,1] 
   }
  }else{
    AR = NA
  }
  return(AR)
}
get_corr<- function(tl, ml){
  m = lm(tl ~ ml + 0)
  s = summary(m)$coefficients
  return(s[1,1])
}
get_data <- function(fn){
  data0 <- fromJSON(txt=fn)
  data = data.frame(prior_los = log_odds(data0$priors),
                    true_los = log_odds(data0$hrms),
                    model_los = log_odds(data0$ams),
                    Ns = data0$weights,
                    sub_exp = data0$sub_exp,
                    thetas = to_theta(data0$lik_ratios)
  )
  
  data$lik_los = data$true_los - data$prior_los
  
  data$lik_los[which(!is.finite(data$lik_los))] = NA
  data$lik_los[is.nan(data$lik_los)] = NA
  
  data$model_los[which(!is.finite(data$model_los))] = NA
  data$model_los[is.nan(data$model_los)] = NA
  return(data)
}
plot_ss <- function(data){
  
  # Ns
  data$random_reassign = sample(c(1, 2, 3), replace = TRUE, size = length(data$Ns))
  d_summary = ddply(data,.(Ns, sub_exp, random_reassign), summarise, get_AR(model_los, lik_los))
  ds = data.frame(Ns = d_summary$Ns,
                  exp = d_summary$sub_exp,
                  ARs = d_summary$..1
  )
  ds = unique(ds)
  
  p <- ggplot(ds, aes(x = Ns, y = ARs, col = exp)) + 
    geom_jitter(width = 1) + xlim(c(-1, 40)) + ylim(c(-0.2, 1.6)) + #geom_smooth(method = 'loess') + 
    # stat_smooth(method = 'lm', se = FALSE)
    stat_smooth(mapping = aes(x = Ns, y = ARs), method = 'lm', inherit.aes = FALSE, col = 'black')
  # geom_abline(intercept = 1.0, slope = 0.0, col = 'red')
  return(p)
}
plot_raw_logodds <- function(data){
  # N = length(data$sub_exp)/1400
  # data$shapes <- rep(c(rep(c(1, 2, 3, 4, 6, 1, 2), each = 200)), N)
  p <- ggplot(data, aes(x = true_los, y = model_los, col = sub_exp)) + 
    geom_point() + xlim(c(-10, 10)) + ylim(c(-4, 4))+ 
    geom_smooth(aes(x = true_los, y = model_los), method = 'loess', inherit.aes = FALSE, col = 'black') + 
    stat_smooth(mapping = aes(x = true_los, y = model_los), method = 'lm', inherit.aes = FALSE, col = 'black') +
    geom_abline(intercept = 0.0, slope = 1.0, col = 'black')
  return(p)
}
plot_logodds <- function(data){
  # data$tl_bins = .bincode(data$true_los, quantile(data$true_los, probs = seq(0.0, 1.0, 0.05)))
  d_summary = ddply(data,.(true_los, sub_exp), summarise, mean_model_los = mean(model_los), mean_true_los = mean(true_los))
  p <- ggplot(d_summary, aes(x = mean_true_los, y = mean_model_los, col = sub_exp)) + 
    geom_point() + xlim(c(-10, 10)) + ylim(c(-4, 4))+ 
    geom_smooth(aes(x = mean_true_los, y = mean_model_los), method = 'loess', inherit.aes = FALSE, col = 'black') + 
    stat_smooth(mapping = aes(x = mean_true_los, y = mean_model_los), method = 'lm', inherit.aes = FALSE, col = 'black', span = 1.0) +
    geom_abline(intercept = 0.0, slope = 1.0, col = 'black')
  return(p)
}
plot_diag <- function(data){
  
  # diagnosticity
  data$random_reassign = sample(c(1, 2, 3, 4), replace = TRUE, size = length(data$thetas))
  d_summary = ddply(data,.(thetas, sub_exp, random_reassign), summarise, get_AR(model_los, lik_los))
  ds = data.frame(thetas = d_summary$thetas,
                  exp = d_summary$sub_exp,
                  ARs = d_summary$..1
  )
  # # diagnosticity
  # data$theta_bins = .bincode(data$thetas, breaks = seq(0, max(data$thetas), 0.1))
  # d_summary = ddply(data,.(theta_bins, sub_exp), summarise, get_AR(model_los, lik_los))
  # ds = data.frame(thetas = d_summary$theta_bins*0.1,
  #                 exp = d_summary$sub_exp,
  #                 ARs = d_summary$..1
  # )
  # 
  
  
  p <- ggplot(ds, aes(x = thetas, y = ARs, col = exp)) + 
    geom_jitter(width = 0.01) + xlim(c(0.55, 0.85)) + ylim(c(-0.5, 1.6)) + #geom_smooth(method = 'loess') + 
    # stat_smooth(method = 'lm')
    stat_smooth(mapping = aes(x = thetas, y = ARs), method = 'lm', inherit.aes = FALSE, col = 'black')
  # geom_abline(intercept = 1.0, slope = 0.0, col = 'red')
  return(p)
}
#***************************
fn = "N_part27__expt_nameBenj__NHID1__NONLINrbf__noise_blocks100__L20.0__test_epoch0__test_lr0.0__train_epoch40__train_lr0.01__plot_data_SS0"
fn = "N_part27__expt_nameBenj__NHID1__NONLINrbf__noise_blocks50__L20.0__test_epoch0__test_lr0.0__train_epoch40__train_lr0.01__plot_data"

data = get_data(fn)
d0 = ddply(data,.(sub_exp), summarise, get_corr(true_los, model_los))
d = data.frame(exp = d0$sub_exp,
               corr = d0$..1
)
p <- ggplot(d, aes(exp, corr)) + geom_col() + ylim(c(-0.2, 1.8))
p
# ggsave(file = "corrs_NHID1.png", p)
# 
# exclude = d$exp[d$corr < 0.0]
exclude = c('SK07', 'KW04')
# exclude = c('PSM65', 'MC72', 'BWB70', 'GHR65', 'DD74')
for (exp in exclude){
  print(exp)
  data = subset(data, sub_exp != exp)
}


p = plot_raw_logodds(data)
p
p = plot_logodds(data)
p
# ggsave(file = "logodds_NHID1.png", p)

p = plot_ss(data)
p
# ggsave(file = "samplesize_NHID1.png", p)

p = plot_diag(data)
p
# ggsave(file = "diagnosticity_NHID1.png", p)

