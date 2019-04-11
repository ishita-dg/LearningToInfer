library(jsonlite)
library(ggplot2)
require(gridExtra)
require(plyr)


setwd("~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}

do_analysis <- function(fn){
  data0 <- fromJSON(txt=fn)
  data = data.frame(prior_los = log_odds(data0$priors),
                    true_los = log_odds(data0$hrms),
                    model_los = log_odds(data0$ams),
                    conds = data0$conds,
                    priors = data0$priors
  )
  data$lik_los = data$true_los - data$prior_los
  data$lpbyll <- data$prior_los/data$lik_los
  
  get_coeffs <- function(model_los, lik_los, prior_los, lpbyll){
    summary = summary(lm(model_los ~ lik_los + prior_los + 0))$coefficients
    if (length(summary) < 8) {
      alpha = mean(model_los/(lik_los+prior_los))
      alpha_l = alpha
      alpha_p = alpha
    }else{
      alpha_l = summary[1,1]
      alpha_p = summary[2,1]
    }
    return(cbind(alpha_l, alpha_p, mean(lpbyll)))
  }
  
  # d_summary = ddply(data,.(conds, priors), .fun = function(df) {cbind(summary(lm(df$model_los ~ df$lik_los + df$prior_los + 0))$coefficients)})
  d_summary = ddply(data,.(conds, priors), summarize, get_coeffs(model_los, lik_los, prior_los, lpbyll))
  ds = data.frame(conds = d_summary$conds,
                  priors = d_summary$priors,
                  alpha_p =  d_summary$..1[1:36, 2],
                  alpha_l =  d_summary$..1[1:36, 1],
                  lpbyll = d_summary$..1[1:36, 3]
  )
  p0 <- ggplot(ds, aes(x = priors, y = alpha_l + (alpha_p - 1)*lpbyll, col = factor(conds))) + 
    geom_line()+
    ylim(c(-0.2, 4))
  return(p0)
}

#***********************************
# Replotting original NHID 1

fn = "N_part49__expt_namePM__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch30__train_lr0.05__plot_data"
plot = do_analysis(fn)
plot
ggsave(file = "PM_NHID1.png", plot)

#***********************************
# Replotting original NHID 5

fn = "N_part49__expt_namePM__NHID5__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch30__train_lr0.05__plot_data"

plot = do_analysis(fn)
ggsave(file = "PM_NHID5.png", plot)


#***********************************
# Plotting new controls -- fixed ll and prior

fn = "N_part49__expt_namePM__fix_llTrue__fix_priorTrue__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch100__train_lr0.02__plot_data"

plot = do_analysis(fn)
ggsave(file = "PM_control.png", plot)

#***********************************
# Plotting new controls -- no training

fn = "N_part49__expt_namePM__fix_llFalse__fix_priorFalse__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch1__train_lr0.05__plot_data"

plot = do_analysis(fn)
ggsave(file = "PM_control_notrain.png", plot)

