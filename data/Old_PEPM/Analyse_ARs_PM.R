library(jsonlite)
library(ggplot2)
require(gridExtra)
require(plyr)


setwd("~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}

Mode <- function(x) {
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}


get_AR <- function(model_los, lik_los, prior_los){
  filter = abs(lik_los) > 0
  # filter = (model_los - prior_los)/lik_los > 0

  summary = summary(lm(model_los[filter] ~ lik_los[filter] + prior_los[filter]+ 0))$coefficients
  if (length(summary) < 8) {
    alpha_l = summary[1,1]
    alpha_p = 0
  }else{
    alpha_l = summary[1,1]
    alpha_p = summary[2,1]
  }
  lpbyll = median(prior_los/lik_los)
  AR = alpha_l + (alpha_p - 1)*lpbyll

  # ARs = ((model_los - prior_los) / lik_los)
  # # ARs = ARs[ARs > 0]
  # AR = median(ARs)
  # # AR = mean(ARs)
  
  # summary = summary(lm(model_los[filter] - prior_los[filter]~ lik_los[filter] + 0))$coefficients
  # AR = summary[1,1]

  # if (prior_los[1] == 0){
  #   ml = model_los[filter]
  #   ll = lik_los[filter]
  # }else{
  #     ml = model_los[filter] - prior_los[filter] #/ prior_los[filter]
  #     ll = lik_los[filter] # / prior_los[filter]
  # }
  # 
  return(AR)
}


do_analysis <- function(fn){
  data0 <- fromJSON(txt=fn)
  data = data.frame(prior_los = log_odds(data0$priors),
                    true_los = log_odds(data0$hrms),
                    model_los = log_odds(data0$ams),
                    # lik_los = -log(data0$lik_ratio),
                    conds = data0$conds,
                    priors = data0$priors
  )
  # which_urn = data$lik_los / abs(data$true_los - data$prior_los)
  # flag = which_urn > 0
  # data$lik_los = data$lik_los*(!flag) + data$lik_los*(flag)
  # data$lik_los = data$true_los - data$prior_los
  
  data$lik_los = data$true_los - data$prior_los
  
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

#***********************************
# Replotting original NHID 1

fn = "N_part49__expt_namePM__fix_llFalse__fix_priorFalse__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch300__train_lr0.02__plot_data"
plot = do_analysis(fn)
plot
# ggsave(file = "PM_NHID1.png", plot)

#***********************************
# Replotting original NHID 5

fn = "N_part49__expt_namePM__fix_llFalse__fix_priorFalse__NHID5__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch100__train_lr0.02__plot_data"

plot = do_analysis(fn)
plot
# ggsave(file = "PM_NHID5.png", plot)


#***********************************
# # Plotting new controls -- fixed ll and prior
# 
# fn = "N_part49__expt_namePM__fix_llTrue__fix_priorTrue__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch100__train_lr0.02__plot_data"
# 
# plot = do_analysis(fn)
# plot
# # ggsave(file = "PM_control.png", plot)
# 
#***********************************
# Plotting new controls -- no training

fn = "N_part49__expt_namePM__fix_llFalse__fix_priorFalse__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch0__train_lr0.02__plot_data"

plot = do_analysis(fn)
plot
# ggsave(file = "PM_control_notrain.png", plot)

