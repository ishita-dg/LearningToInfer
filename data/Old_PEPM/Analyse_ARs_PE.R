library(jsonlite)
library(ggplot2)
require(gridExtra)
require(plyr)


setwd("~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x) - log((1.0 - x)))
}

log_odds_exp <- function(x){
  return(x - log((1.0 - exp(x))))
}


get_AR <- function(model_los, lik_los){
  summary = summary(lm(model_los ~ lik_los + 0))$coefficients
  if (length(summary) < 4){
    AR = 1.0
  }else{
    AR = summary[1,1]
  }
  if (length(lik_los) < L){
    AR = NA
  }
  # AR = median(model_los/lik_los)
  # print(length(lik_los))
  # print(AR)
  # print("****")
  return(AR)
}

do_analysis <- function(fn){
  data0 <- fromJSON(txt=fn)
  data = data.frame(Ns = data0$Ns,
                    true_los = log_odds(data0$hrms),
                    model_los = log_odds(data0$ams),
                    # lik_los = -log(data0$liks),
                    conds = data0$conds,
                    priors = data0$priors
  )
  
  data$lik_los = data$true_los
  data$lik_los[is.infinite(data$lik_los)] = 100000
  data$model_los[is.infinite(data$model_los)] = 100000
  
  d_summary = ddply(data,.(conds, Ns), summarise, get_AR(model_los, lik_los))
  size = nrow(d_summary$..1)
  ds = data.frame(conds = d_summary$conds,
                  Ns = d_summary$Ns,
                  ARs = d_summary$..1
  )
  ds = unique(ds)
  p0 <- ggplot(ds, aes(x = Ns*20, y = ARs, col = factor(conds))) + 
    geom_line()+
    ylim(c(-0.5, 2))+
    xlim(c(-5, 16))+
    ggtitle(fn)
    # geom_line()
  return(p0)
}

#***********************************
# Replotting original NHID 1
# 
# files = list.files(path = ".")
# # file = files[lapply(files, function(strs, pattern) grepl(pattern, strs), pattern = "PM")]
# for (f in files){
#   if (grepl('expt_namePE__fix_ll', f)) {
#     fn <- f
#     print(fn)
#     plot = do_analysis(fn)
#     print(plot)
#     
#   }
# }
fn = "N_part49__expt_namePE__fix_llFalse__fix_priorFalse__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch100__train_lr0.02__plot_data"
L = 1000
plot = do_analysis(fn)
plot
# ggsave(file = "PE_NHID1.png", plot)

#***********************************
# Replotting original NHID 5

fn = "N_part49__expt_namePE__fix_llFalse__fix_priorFalse__NHID5__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch100__train_lr0.02__plot_data"
L = 500
plot = do_analysis(fn)
plot
# ggsave(file = "PE_NHID5.png", plot)

#***********************************
# Random

fn = "N_part49__expt_namePE__fix_llFalse__fix_priorFalse__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch0__train_lr0.02__plot_data"
plot = do_analysis(fn)
plot
ggsave(file = "PE_random.png", plot)



