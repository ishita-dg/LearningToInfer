library(jsonlite)
library(ggplot2)
require(gridExtra)
setwd("~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}

priors = rep(seq(0.01, 0.99, 0.005), times = 197)
likls = rep(seq(0.01, 0.99, 0.005), each = 197)

prior_los <- log_odds(priors)
lik_los <- log_odds(likls)
true_los <- prior_los + lik_los

# lik_bins = .bincode(lik_los, breaks = quantile(lik_los, seq(0, 1, 0.5)))
# pri_bins = .bincode(prior_los, breaks = quantile(prior_los, seq(0, 1, 0.5)))
# 
# xlik_los = lik_los
# xpri_los = prior_los
# for (bin in c(1,2)){
#   xlik_los[lik_bins == bin] <- mean(lik_los[lik_bins == bin], na.rm = TRUE)
#   xpri_los[pri_bins == bin] <- mean(prior_los[pri_bins == bin], na.rm = TRUE)
# }
# model_los <- xlik_los + xpri_los



# one_bit <- true_los
# one_bit[bit_bins == 1] <- log_odds(mean(hrms[bit_bins == 1], na.rm = TRUE))
# one_bit[bit_bins == 2] <- log_odds(mean(hrms[bit_bins == 2], na.rm = TRUE))
# 
# # one_bit[bit_bins == 1] <- mean(true_los[bit_bins == 1], na.rm = TRUE)
# # one_bit[bit_bins == 2] <- mean(true_los[bit_bins == 2], na.rm = TRUE)
# 
# model_los = one_bit
# # Bin the lik_los


plot_cuts = seq(0, 1, 0.25)
df_final = data.frame(lik = c(),  
                      coeff = c(),
                      label = c())
for (cut_width in c(0.02, 0.1, 0.25, 0.3, 0.5)){
  
  cuts = seq(0, 1, cut_width)
  val_bins = .bincode(true_los, breaks = quantile(true_los, cuts), right = TRUE, include.lowest = TRUE)
  model_los <- true_los
  for (bin in seq(1,length(cuts) - 1)){
    model_los[val_bins == bin] <- mean(model_los[val_bins == bin], na.rm = TRUE)
  }
  
  bins = .bincode(abs(lik_los), breaks = quantile(abs(lik_los), plot_cuts), right = TRUE, include.lowest = TRUE)
  
  coeffs = c()
  for (bin in seq(1,length(plot_cuts) - 1)){
    model_lm0 <- lm(model_los[bins == bin] ~ lik_los[bins == bin] + prior_los[bins == bin] + 0)
    coeffs[bin] = model_lm0$coefficients[1]
  }
  x = (quantile(abs(lik_los), plot_cuts)[2:length(plot_cuts)] + quantile(abs(lik_los), plot_cuts)[1:length(plot_cuts) - 1])/2.0
  print(rep(paste(length(cuts) - 1, "bins", sep = ""), length(x)))
  df = data.frame(lik = x,
                  coeff = coeffs,
                  label = rep(paste(length(cuts) - 1, "bins", sep = ""), length(x)))
  df_final = rbind(df_final, df)
}

plot1 <- ggplot(df_final, aes(y = coeff, x = lik, color = label)) +
  geom_hline(yintercept=1.0, linetype="dashed", color = "black") +
  geom_point() + geom_line() + ggtitle("System Neglect with changing bins")  + ylim(c(0.5, 1.05))
ggsave(file ="Intuition_allbins.pdf", plot1)
plot1
  
# # Prior bins
# bins = .bincode(abs(prior_los), breaks = quantile(abs(prior_los), plot_cuts), right = TRUE, include.lowest = TRUE)
# 
# coeffs = c()
# for (bin in seq(1,length(plot_cuts) - 1)){
#   model_lm0 <- lm(model_los[bins == bin] ~ lik_los[bins == bin] + prior_los[bins == bin] + 0)
#   coeffs[bin] = model_lm0$coefficients[2]
# }
# 
# df = data.frame(prior = (quantile(abs(prior_los), plot_cuts)[2:length(plot_cuts)] + quantile(abs(prior_los), plot_cuts)[1:length(plot_cuts) - 1])/2.0,  
#                 coeff = coeffs)
# plot2 <- ggplot(df, aes(y = coeff, x = prior)) +
#   geom_point() + geom_line() + ggtitle("Prior binned regression coeff")  + ylim(c(0.4, 1.2))
# plot2
# 
# 
# All plots together


 
