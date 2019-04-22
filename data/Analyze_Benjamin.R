library(jsonlite)
library(ggplot2)
require(gridExtra)
require(plyr)
library(latex2exp)


setwd(" ~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}

to_theta <- function(x) {
  t = x/(1.0 + x)
  return(0.5 + abs(t-0.5))
}
get_AR <- function(model_los, lik_los){
  # print(sum(model_los %in% boxplot.stats(model_los)$out))
  # model_los[model_los %in% boxplot.stats(model_los)$out] = NA
  if (sum(is.na(model_los)) < length(model_los)){
   summary = summary(lm(model_los ~  lik_los + 0, na.action=na.exclude))$coefficients 
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
  m = lm(tl ~  ml + 0)
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
  data$model_los[data$model_los > 17] = NA
  data$model_los[data$model_los < -17] = NA
  return(data)
}
plot_ss <- function(data, N = 3){
  
  # Ns
  data$random_reassign = sample(seq(1, N), replace = TRUE, size = length(data$Ns))
  d_summary = ddply(data,.(Ns, sub_exp, random_reassign), summarise, get_AR(model_los, lik_los))
  ds = data.frame(Ns = d_summary$Ns,
                  exp = d_summary$sub_exp,
                  ARs = d_summary$..1
  )
  ds = unique(ds)
  # ds$exp_group <- mapvalues(ds$exp, from = all_exps, 
                            # to = map_all_exps)
  
  
  p <- ggplot(ds, aes(x = Ns, y = ARs, col = exp, shape = exp)) + 
    theme_classic() + xlim(c(-1, 50)) + ylim(c(-0.5, 3.0)) + #
    geom_smooth(mapping = aes(x = Ns, y = ARs), method = 'loess',  linetype="dotdash", 
    inherit.aes = FALSE, col = 'black', span = 1.75) +
    xlab(TeX("$N$"))+
    ylab(TeX("$\\hat{\\alpha_L} = \\frac{log   \\frac{\\pi(A| d)}{\\pi(B| d)}}{log   \\frac{P(A| d)}{P(B| d)}}$"))+
    # stat_smooth(method = 'lm', se = FALSE)
    theme(legend.position = 'none', legend.text=element_text(size=8), axis.text=element_text(size=10),
          axis.title=element_text(size=10),
          axis.title.y = element_text(size = 10, margin = margin(t = 0, r = -15, b = 0, l = 0))) +
    geom_abline(intercept = 1.0, slope = 0.0,  linetype="dotted",  col = 'black') +
    stat_smooth(mapping = aes(x = Ns, y = ARs), method = 'lm', 
                inherit.aes = FALSE, col = 'black') + geom_jitter( size = 3, width = 0.5)+
    scale_shape_manual("", values=map_all_exps) + scale_colour_discrete("") 
  # theme(legend.position = 'none', legend.text=element_text(size=14), axis.text=element_text(size=10),         axis.title=element_text(size=10)) + geom_abline(intercept = 1.0, slope = 0.0, col = 'red')
  return(p)
}
plot_raw_logodds <- function(data, distinct = FALSE, Nbins = 30){
  
  data = tail(data, 10000)
  p <- ggplot(data, aes(x = true_los, y = model_los))+#, col = sub_exp, shape = sub_exp)) + 
    theme_classic() + xlim(c(-15, 15)) + ylim(c(-15, 15))+ 
    geom_point( size = 0.1 )+
    xlab(TeX("$log   \\frac{P(A| d)}{P(B| d)}$"))+
    ylab(TeX("$log  \\frac{\\pi(A| d)}{\\pi(B| d)}$"))+
    # geom_line(aes(x = sort(true_los), y = log_odds(seq(0.0001, 1.0, 0.0001))), inherit.aes = FALSE, col = 'red') + 
    geom_smooth(aes(x = true_los, y = model_los), method = 'loess',  linetype="dotdash", 
                inherit.aes = FALSE, col = 'blue', span = 1.0) + 
    # stat_smooth(mapping = aes(x = true_los, y = model_los), method = 'lm', 
    #             inherit.aes = FALSE, col = 'black') +
    theme(legend.position = 'none', legend.text=element_text(size=14), 
          axis.title.y = element_text(size = 10, margin = margin(t = 0, r = -15, b = 0, l = 0)), 
          axis.title=element_text(size=10)) + 
    geom_abline(intercept = 0.0, slope = 1.0,  linetype="dotted",  col = 'black') + 
    scale_shape_manual("", values=map_all_exps) + scale_colour_discrete("")
  p0<-ggExtra::ggMarginal(p, type = "histogram", bins = Nbins)
  return(p0)
}
plot_logodds <- function(data, N = 3){
  # data$tl_bins = .bincode(data$true_los, quantile(data$true_los, probs = seq(0.0, 1.0, 0.05)))
  data$random_reassign = sample(seq(1, N), replace = TRUE, size = length(data$true_los))
  d_summary = ddply(data,.(true_los, sub_exp, random_reassign), summarise, 
                    mean_model_los = median(model_los, na.rm = TRUE), mean_true_los = median(true_los, na.rm = TRUE))
  # d_summary$int <- paste(d_summary$sub_exp, d_summary$exp_group, sep=".")
  p <- ggplot(d_summary, aes(x = mean_true_los, y = mean_model_los, col = sub_exp, shape = sub_exp)) + 
    theme_classic() + xlim(c(-10, 15)) + ylim(c(-7, 7))+
    xlab(TeX("$log   \\frac{P(A| d)}{P(B| d)}$"))+
    ylab(TeX("$log  \\frac{\\pi(A| d)}{\\pi(B| d)}$"))+
    geom_smooth(aes(x = mean_true_los, y = mean_model_los), method = 'loess',  linetype="dotdash", 
                inherit.aes = FALSE, col = 'black') + 
    stat_smooth(mapping = aes(x = mean_true_los, y = mean_model_los), method = 'lm', 
                inherit.aes = FALSE, col = 'black', span = 1.75) +
    theme(legend.position = 'none', legend.text=element_text(size=14), 
          axis.text=element_text(size=10),         
          axis.title=element_text(size=10),
          axis.title.y = element_text(size = 10, margin = margin(t = 0, r = -15, b = 0, l = 0))) + 
    geom_abline(intercept = 0.0, slope = 1.0,  linetype="dotted",  col = 'black') + geom_point( size = 3, )+
    scale_shape_manual("", values=map_all_exps) + scale_colour_discrete("")
  return(p)
}

plot_priorlogodds <- function(data, N = 3, raw = TRUE){
  
  # diagnosticity
  data$random_reassign = sample(seq(1, N), replace = TRUE, size = length(data$thetas))
  d_summary = ddply(data,.(sub_exp, random_reassign), summarise, get_AR(model_los, lik_los))
  ds = data.frame(exp = d_summary$sub_exp,
                  ARs = d_summary$..1,
                  group = d_summary$random_reassign
  )
  data$int <- paste(data$sub_exp, data$random_reassign, sep=".")
  data$ARs <- as.integer(mapvalues(data$int, from = paste(ds$exp, ds$group, sep="."), 
                              to = ds$ARs))
  data$adjusted_los <- data$model_los - data$ARs*data$lik_los
  
  data$random_reassign = sample(seq(1, N), replace = TRUE, size = length(data$true_los))
  d_summary = ddply(data,.(prior_los, sub_exp, random_reassign), summarise, 
                    mean_prior_los = median(prior_los, na.rm = TRUE), mean_adjusted_los = median(adjusted_los, na.rm = TRUE))
  p <- ggplot(d_summary, aes(x = mean_prior_los, y = mean_adjusted_los, col = sub_exp, shape = sub_exp)) + 
    theme_classic() + xlim(c(-2.1, 2.1)) + ylim(c(-3, 3))+ coord_fixed(ratio = 0.5) +
    geom_jitter(size = 3, width = 0.05) +
    xlab(TeX("$log   \\frac{P(A)}{P(B)}$"))+
    ylab(TeX("$log  \\frac{\\pi(A | d)}{\\pi(B| d)} - \\hat{\\alpha_L} \\frac{\\P(d | A)}{\\P(d| B)}$"))+
    stat_smooth(mapping = aes(x = mean_prior_los, y = mean_adjusted_los), method = 'lm', 
                inherit.aes = FALSE, col = 'black') +
    # stat_smooth(mapping = aes(x = mean_prior_los, y = mean_adjusted_los), method = 'loess', 
    #             inherit.aes = FALSE, linetype="dotdash", col = 'black', span = 1.5) +
    geom_abline(intercept = 0.0, slope = 1.0,  linetype="dotted",  col = 'black') + 
    theme(legend.position = 'none', legend.text=element_text(size=8), 
          axis.text=element_text(size=8),         
          axis.title=element_text(size=8),
          axis.title.y = element_text(size = 8, margin = margin(t = 0, r = -15, b = 0, l = 0))) + 
    scale_shape_manual("", values=map_all_exps) + scale_colour_discrete("")
  
  if (raw){
    return(p)
  }
  data$random_reassign = sample(seq(1, 1), replace = TRUE, size = length(data$thetas))
  d_summary0 = ddply(data,.(prior_los, sub_exp, random_reassign), summarise, 
                    mean_prior_los = median(model_los, na.rm = TRUE), mean_adjusted_los = median(adjusted_ll, na.rm = TRUE))
  
  p <- ggplot(d_summary0, aes(x = mean_prior_los, y = mean_adjusted_los, col = sub_exp, shape = sub_exp)) + 
    theme_classic() + xlim(c(-15, 15)) + ylim(c(-7, 7))+
    stat_smooth(mapping = aes(x = mean_prior_los, y = mean_adjusted_los), method = 'lm', 
                inherit.aes = FALSE, col = 'red') +
    theme(legend.position = 'none', legend.text=element_text(size=14), axis.text=element_text(size=10),         axis.title=element_text(size=10)) + geom_abline(intercept = 0.0, slope = 1.0,  linetype="dotted",  col = 'black') + geom_point( size = 3 )+
    scale_shape_manual("", values=map_all_exps) + scale_colour_discrete("")
  return(p)
}


plot_diag <- function(data, N = 3){
  
  # diagnosticity
  data$random_reassign = sample(seq(1, N), replace = TRUE, size = length(data$thetas))
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
  
  
  p <- ggplot(ds, aes(x = thetas, y = ARs, col = exp, shape = exp)) + 
     theme_classic() + xlim(c(0.55, 0.85)) + ylim(c(-0.1, 1.5)) + #geom_smooth(method = 'loess',  linetype="dotdash") +
     # stat_smooth(method = 'lm')
    xlab(TeX("$\\theta$"))+
    ylab(TeX("$\\hat{\\alpha_L} = \\frac{log\\frac{\\pi(A| d)}{\\pi(B| d)}}{log\\frac{P(A| d)}{P(B| d)}}$"))+
    stat_smooth(mapping = aes(x = thetas, y = ARs), method = 'lm', 
                inherit.aes = FALSE, col = 'black', span = 1.75) + 
    theme(legend.position = 'none', legend.text=element_text(size=14), 
          axis.text=element_text(size=10),         
          axis.title=element_text(size=10),
          axis.title.y = element_text(size = 10, margin = margin(t = 0, r = -15, b = 0, l = 0))) + 
    geom_abline(intercept = 1.0, slope = 0.0,  linetype="dotted",  col = 'black') +
    geom_jitter( size = 4, width = 0.01)+
    scale_shape_manual("", values=map_all_exps) + scale_colour_discrete("")
  # theme(legend.position = 'none', legend.text=element_text(size=14), axis.text=element_text(size=10),         axis.title=element_text(size=10)) + geom_abline(intercept = 1.0, slope = 0.0, col = 'red')
  return(p)
}
#***************************
hidden = 8
type = 'prior'
setting = paste(hidden, type, sep = '')
if (hidden == 2){
  if (type == 'notrain'){
    fn = "N_part299__diff_noiseTrue__expt_nameBenj__NHID2__NONLINrbf__noise_blocks150__L20.0__test_epoch0__test_lr0.0__train_epoch0__train_lr0.01__train_blocks150__plot_data"
    }
  else if (type == 'biased'){
    fn = "N_part149__diff_noiseTrue__expt_nameBenj__NHID2__NONLINrbf__noise_blocks150__L20.0__test_epoch0__test_lr0.0__train_epoch50__train_lr0.01__query_manipTrue__train_blocks150__unif_samples_plot_data"
  }
  else if(type == 'prior'){
    fn = "N_part179__diff_noiseFalse__expt_nameBenj_prior__NHID2__NONLINrbf__noise_blocks150__L20.0__test_epoch0__test_lr0.0__train_epoch60__train_lr0.01__train_blocks150__plot_data"
    # fn = "N_part79__diff_noiseTrue__expt_nameBenj_prior__NHID1__NONLINrbf__noise_blocks150__L20.0__test_epoch0__test_lr0.0__train_epoch50__train_lr0.01__train_blocks150__plot_data"
  }
  else{
    fn = "N_part299__expt_nameBenj__NHID2__NONLINrbf__noise_blocks150__L20.0__test_epoch0__test_lr0.0__train_epoch50__train_lr0.01__train_blocks150__plot_data"
  }
}
if(hidden == 1){
  fn = "N_part299__diff_noiseTrue__expt_nameBenj__NHID1__NONLINrbf__noise_blocks150__L20.0__test_epoch0__test_lr0.0__train_epoch50__train_lr0.01__train_blocks150__plot_data"
  }
if(hidden == 5){
  fn = "N_part299__expt_nameBenj__NHID5__NONLINrbf__noise_blocks150__L20.0__test_epoch0__test_lr0.0__train_epoch50__train_lr0.01__train_blocks150__plot_data"
  }
if(hidden == 8){
  if(type == 'prior'){
    fn = "N_part119__diff_noiseFalse__expt_nameBenj_prior__NHID8__NONLINrbf__noise_blocks150__L20.0__test_epoch0__test_lr0.0__train_epoch200__train_lr0.01__train_blocks150__plot_data"
  }else {
    fn = "N_part299__diff_noiseTrue__expt_nameBenj__NHID8__NONLINrbf__noise_blocks150__L20.0__test_epoch0__test_lr0.0__train_epoch200__train_lr0.01__train_blocks150__plot_data"
  }
  }

data = get_data(fn)

exclude = c('BH80')
for (exp in exclude){
  print(exp)
  data = subset(data, sub_exp != exp)
}


all_exps = c('BH80','BWB70', 'DD74', 'GHR65', 'GHR65-p', 'Gr92', 'GT92', 'GT92-p', 'HS09', 
             'KW04', 'MC72', 'Ne01', 'PM65', 'PSM65', 'SK07')
if (setting == 'prior'){
  map_all_exps = seq(1, length(unique(data$sub_exp)))
  # map_all_exps = c(2, 1, 4, 10, 18, 15)
  }else {
    map_all_exps = c(1, 20, 15, 18, 4, 15, 10, 0, 17, 6, 2, 5, 18)
  }
map_exp_names = c('Bar-Hillel (1980)', 'Beach, Wise & Barclay (1970)', 
                  'Donnell & Du Charme (1974)', 'Green, Halbert, & Robinson (1965)', 
                  'Green, Halbert, & Robinson (1965)','Grether (1992)', 'Griffin & Tversky (1992)',
                  'Griffin & Tversky (1992)', 'Holt & Smith (2009)', 
                  'Kraemer & Weber (2004)', 'Marks and Clarkson (1972)', 'Nelson et al (2001)', 
                  'Peterson & Miller (1965)', 'Peterson, Schneider, & Miller (1965)', 
                  'Sasaki & Kawagoe (2007)')

data$sub_exp <- mapvalues(data$sub_exp, from = all_exps,
                          to = map_exp_names)
# all_exps = sort(unique(data$sub_exp))
# map_all_exps = seq(1, length(all_exps))
# map_all_exps[7:13] = 1:7
# map_all_exps[1:5] = 16:20
# map_all_exps[12] = 1
# map_all_exps[1] = 18
# map_all_exps[11] = 23

d0 = ddply(data,.(sub_exp), summarise, get_corr(true_los, model_los))
d = data.frame(exp = d0$sub_exp,
               corr = d0$..1
)
p <- ggplot(d, aes(exp, corr)) + geom_col() + ylim(c(-0.2, 1.8))
p
# ggsave(file = "corrs_NHID1.png", p)
# 
# exclude = d$exp[d$corr < 0.0]

# p = plot_raw_logodds(data)
# ggsave(file = paste("raw_logodds_NHID", setting, ".png", sep = ''), p)


if (type == 'prior'){
  p = plot_priorlogodds(data, N = 5)
  p
  # ggsave(file = paste("legend_", setting, ".pdf", sep = ''), p)
  ggsave(file = paste("prior_NHID", setting, ".png", sep = ''), p)
} else {
  
  p = plot_logodds(data, N = 1)
  p
  ggsave(file = paste("logodds_NHID", setting, ".png", sep = ''), p)
  
  p = plot_ss(data, N = 3)
  p
  ggsave(file = paste("samplesize_NHID", setting, ".png", sep = ''), p)
  
  p = plot_diag(data, N = 3)
  p
  ggsave(file = paste("diagnosticity_NHID", setting, ".png", sep = ''), p)
  
}

# data0 <- as.data.frame(fromJSON(txt=fn))
# data0$hrm_bins = .bincode(data0$hrms, breaks = seq(0.01, 1.0, 0.01))
# d_sum <- ddply(data0,.(hrm_bins), summarise, mean_am = mean(ams), mean_hrm = mean(hrms))
# N = length(d_sum$mean_am)
# p <- ggplot(d_sum, aes(x = mean_am, y = mean_hrm)) + geom_point() + geom_smooth(method = 'loess',  linetype="dotdash", col = "black")+
#   geom_smooth(mapping = aes(x = sort(mean_am), y = seq(1.0/N, 1.0, 1.0/N)), 
#               inherit.aes = FALSE, method = 'loess',  linetype="dotdash")
# p
# p <- ggplot(data0, aes(x = sort(hrms),y = seq(1.0/N, 1.0, 1.0/N)) ) + geom_smooth( method = 'loess',  linetype="dotdash")
# p

