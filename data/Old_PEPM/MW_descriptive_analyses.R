library(jsonlite)
library(ggplot2)
require(gridExtra)
require(plyr)


setwd("~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}

get_AR <- function(model_los, lik_los, prior_los){
  summary = summary(lm(model_los ~ lik_los + prior_los+ 0))$coefficients
  if (length(summary) < 8) {
    alpha = mean(model_los/(lik_los+prior_los))
    alpha_l = alpha
    alpha_p = alpha
  }else{
    alpha_l = summary[1,1]
    alpha_p = summary[2,1]
  }
  AR = alpha_l + (alpha_p - 1)*mean((prior_los/lik_los))
  # print(lik_los[1])
  # print(prior_los[1])
  print(summary)
  
  # AR = summary[2,1]
  return(AR)
}

get_coeffs <- function(model_los, lik_los, prior_los){
  summary = summary(lm(model_los ~ lik_los + prior_los + 0))$coefficients
  if (length(summary) < 8) {
    alpha = mean(model_los/(lik_los+prior_los))
    alpha_l = alpha
    alpha_p = alpha
  }else{
    alpha_l = summary[1,1]
    alpha_p = summary[2,1]
  }
  print(summary)
  print(alpha_l)
  print(alpha_p)
  return(cbind(alpha_l, alpha_p))
}


do_analysis <- function(fn){
  data0 <- fromJSON(txt=fn)
  data = data.frame(prior_los = log_odds(data0$priors),
                    true_los = log_odds(data0$hrms),
                    model_los = log_odds(data0$ams),
                    # lik_los = -log(data0$liks),
                    conds = data0$conds,
                    priors = data0$priors
  )
  data$lik_los = data$true_los - data$prior_los
  data$lik_los[is.infinite(data$lik_los)] = 1000
  data$prior_cond = round(abs(data$prior_los), digits = 2)
  # data$model_los[is.infinite(data$model_los)] = 1000
  # # which_urn = data$lik_los / abs(data$true_los - data$prior_los)
  # flag = which_urn > 0
  # data$lik_los = data$lik_los*(!flag) + data$lik_los*(flag)
  # data$lik_los = data$true_los - data$prior_los
  data$lpbyll <- data$prior_los/data$lik_los
  
  # data$prior_bins = .bincode(priors, breaks = c(0.0, 0.25, 0.48, 0.52, 0.75,))
  
  
  # d_summary = ddply(data,.(prior_cond, conds), summarise, get_coeffs(model_los, lik_los, prior_los))
  d_s_p = ddply(data,.(prior_cond), summarise, get_coeffs(model_los, lik_los, prior_los))
  d_s_l = ddply(data,.(conds), summarise, get_coeffs(model_los, lik_los, prior_los))
  # ds = unique(ds)
  return(ds)
}

#***********************************
# Replotting original NHID 1

fn = "N_part4__expt_nameMW__fix_llFalse__fix_priorFalse__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch40__train_lr0.02__plot_data"
ds = do_analysis(fn)
p1 <- ggplot(ds, aes(x = conds, y = alpha_l)) + 
  geom_point()+
  geom_line()
p1
p2 <- ggplot(ds, aes(x = priors, y = alpha_p)) + 
  geom_point()+
  geom_line()
p2
# ggsave(file = "PM_NHID1.png", plot)

