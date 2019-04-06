library(jsonlite)
library(ggplot2)
require(gridExtra)

Nbins = 18
log_odds <- function(x){
  return(log(x/(1.0 - x)))
}


subsample_unif <- function(samples, bins){
  h = hist(samples, bins)
  thresh = min(h$density)
  bins = .bincode(samples, breaks = h$breaks)
  accept_prob = 1.0 - (h$density[bins] - thresh)/h$density[bins]
  return (lapply(accept_prob, function(p) rbinom(1, 1, p)) > -0.5)
  # return (lapply(accept_prob, function(p) rbinom(1, 1, p)) > 0.5)
}
make_sn_df<- function(plos, llos, mlos){
  
  bins = .bincode(abs(llos), breaks = quantile(abs(llos)))
  
  model_lm0 <- lm(mlos[bins == 1] ~ llos[bins == 1] + plos[bins == 1] + 0)
  
  model_lm1 <- lm(mlos[bins == 2] ~ llos[bins == 2] + plos[bins == 2] + 0)
  
  model_lm2 <- lm(mlos[bins == 3] ~ llos[bins == 3] + plos[bins == 3] + 0)
  
  model_lm3 <- lm(mlos[bins == 4] ~ llos[bins == 4] + plos[bins == 4] + 0)
  
  df0 = data.frame(x = (quantile(abs(llos))[2:5] + quantile(abs(llos))[1:4])/2.0,  
                   coeff = c(model_lm0$coefficients[1], model_lm1$coefficients[1], model_lm2$coefficients[1], model_lm3$coefficients[1]),
                   source = c(rep("Likelihood", 4)))
  
  # Prior bins
  bins = .bincode(abs(plos), breaks = quantile(abs(plos)))
  
  model_lm0 <- lm(mlos[bins == 1] ~ plos[bins == 1] + llos[bins == 1] + 0)
  
  model_lm1 <- lm(mlos[bins == 2] ~ plos[bins == 2] + llos[bins == 2] + 0)
  
  model_lm2 <- lm(mlos[bins == 3] ~ plos[bins == 3] + llos[bins == 3] + 0)
  
  model_lm3 <- lm(mlos[bins == 4] ~ plos[bins == 4] + llos[bins == 4] + 0)
  
  df1 = data.frame(x = (quantile(abs(plos))[2:5] + quantile(abs(plos))[1:4])/2.0,  
                   coeff = c(model_lm0$coefficients[1], model_lm1$coefficients[1], model_lm2$coefficients[1], model_lm3$coefficients[1]),
                   source = c(rep("Prior", 4)))
  df = rbind(df0, df1)
  return(df)
}

make_sn_plot<- function(fn){
  data <- fromJSON(txt=fn)
  prior_los <- log_odds(data$priors)
  true_los <- log_odds(data$hrms)
  lik_los <- true_los - prior_los
  model_los <- log_odds(data$ams)
  
  
  df_sn <- make_sn_df(prior_los, lik_los, model_los)
  plot <- ggplot(df_sn, aes(y = coeff, x = x, color = source, linetype = source)) +
    geom_abline(slope = 0, linetype="dotted", color = "black",  intercept = 1.0) +
    geom_abline(slope = 0, linetype="dotted", color = "black",  intercept = 0.0) +
    geom_line(size=1.1)+  #0 to 1
    scale_y_continuous(expand = c(0,0), limits = c(-0.5,1.5))+  #golden ratio error bars
    #geom_errorbar(limits, position="dodge", width=0.05)+
    #point size
    geom_point(size=1.5)+
    theme_classic() +
    ylab("Regression Coefficient")+xlab("Absolute log odds")+
    #adjust text size
    theme(text = element_text(size=16, family="sans"))+
    theme(panel.background = element_blank(),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          axis.line = element_line(colour = "black"),
          legend.position = "bottom")
  return(plot)
}

make_hist_plot <- function(fn){
  data <- fromJSON(txt=fn)
  
  true0 <- data$hrms
  model0 <- data$ams
  
  sub_index = subsample_unif(true0, Nbins)
  true <- log_odds(true0[sub_index])
  model <- log_odds(model0[sub_index])
  
  df = data.frame(x = c(true, true),
                  y = c(true, model),
                  kind = c(rep("True", length(true)), rep("model", length(model))))
  
  
  df = data.frame(x = true,
                  y = model)
  df$cut_true <- cut(df$x, breaks = seq(-5, 5, 0.4))
  
  p <- ggplot(df,  aes(y = y, x = x))+ #factor(cut_true))) +
    # geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
    geom_point(size = 0.001)+
    geom_smooth(method = 'loess')+
    ylab("Approximate posterior log-odds") + 
    xlab("True posterior log-odds") + 
    ylim(c(-5, 5)) +
    xlim(c(-5, 5)) +
    geom_abline(slope - 1.0, intercept = 0.0) +
    theme_classic() +
    theme(axis.text.x = element_blank(), text = element_text(size=18)) 
  p0<-ggExtra::ggMarginal(p, type = "histogram", bins = Nbins)
  return (p0)
}
setwd("~/GitHub/LearningToInfer/data")

fn = "BiasFalse__N_part19__expt_nameSN__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_blocks400__train_epoch100__train_lr0.05__plot_data"
p0 = make_hist_plot(fn)
p0
ggsave("Demo_low.png", p0)

p0 = make_sn_plot(fn)
p0
ggsave("Demo_low_SN.png", p0)


# *****************

fn = "BiasFalse__N_part19__expt_nameSN__NHID5__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_blocks400__train_epoch100__train_lr0.05__plot_data"
p0 = make_hist_plot(fn)
ggsave("Demo_high.png", p0)

p0 = make_sn_plot(fn)
p0
ggsave("Demo_high_SN.png", p0)


#*************************

fn = "BiasFalse__N_part19__expt_nameSN__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_blocks400__train_epoch0__train_lr0.05__plot_data"
p0 = make_hist_plot(fn)
ggsave("Demo_nolearning.png", p0)

p0 = make_sn_plot(fn)
p0
ggsave("Demo_nolearning_SN.png", p0)

# *****************

fn = "BiasTrue__N_part19__expt_nameSN__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_blocks400__train_epoch100__train_lr0.05__plot_data"
p0 = make_hist_plot(fn)
p0
ggsave("Demo_skewed.png", p0)

p0 = make_sn_plot(fn)
p0
ggsave("Demo_skewed_SN.png", p0)

