library(jsonlite)
library(ggplot2)
require(gridExtra)
setwd("~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}

priors = runif(5000)
likls = runif(5000)

prior_los <- log_odds(priors)
lik_los <- log_odds(likls)
true_los <- prior_los + lik_los

lik_bins = .bincode(lik_los, breaks = quantile(lik_los, seq(0, 1, 0.5)))
pri_bins = .bincode(prior_los, breaks = quantile(prior_los, seq(0, 1, 0.5)))

xlik_los = lik_los
xpri_los = prior_los
for (bin in c(1,2)){
  xlik_los[lik_bins == bin] <- mean(lik_los[lik_bins == bin], na.rm = TRUE)
  xpri_los[pri_bins == bin] <- mean(prior_los[pri_bins == bin], na.rm = TRUE)
}
model_los <- xlik_los + xpri_los


# one_bit <- true_los
# one_bit[bit_bins == 1] <- log_odds(mean(hrms[bit_bins == 1], na.rm = TRUE))
# one_bit[bit_bins == 2] <- log_odds(mean(hrms[bit_bins == 2], na.rm = TRUE))
# 
# # one_bit[bit_bins == 1] <- mean(true_los[bit_bins == 1], na.rm = TRUE)
# # one_bit[bit_bins == 2] <- mean(true_los[bit_bins == 2], na.rm = TRUE)
# 
# model_los = one_bit
# # Bin the lik_los

bins = .bincode(abs(lik_los), breaks = quantile(abs(lik_los)))

model_lm0 <- lm(model_los[bins == 1] ~ lik_los[bins == 1] + prior_los[bins == 1] + 0)
summary(model_lm0)

model_lm1 <- lm(model_los[bins == 2] ~ lik_los[bins == 2] + prior_los[bins == 2] + 0)
summary(model_lm1)

model_lm2 <- lm(model_los[bins == 3] ~ lik_los[bins == 3] + prior_los[bins == 3] + 0)
summary(model_lm2)

model_lm3 <- lm(model_los[bins == 4] ~ lik_los[bins == 4] + prior_los[bins == 4] + 0)
summary(model_lm3)

df = data.frame(lik = (quantile(abs(lik_los))[2:5] + quantile(abs(lik_los))[1:4])/2.0,  
                coeff = c(model_lm0$coefficients[1], model_lm1$coefficients[1], model_lm2$coefficients[1], model_lm3$coefficients[1]))
plot1 <- ggplot(df, aes(y = coeff, x = lik)) +
  geom_point() + geom_line() + ggtitle("Lik binned regression coeff") 
plot1

# Prior bins
bins = .bincode(abs(prior_los), breaks = quantile(abs(prior_los)))

model_lm0 <- lm(model_los[bins == 1] ~ prior_los[bins == 1] + lik_los[bins == 1] + 0)
summary(model_lm0)

model_lm1 <- lm(model_los[bins == 2] ~ prior_los[bins == 2] + lik_los[bins == 2] + 0)
summary(model_lm1)

model_lm2 <- lm(model_los[bins == 3] ~ prior_los[bins == 3] + lik_los[bins == 3] + 0)
summary(model_lm2)

model_lm3 <- lm(model_los[bins == 4] ~ prior_los[bins == 4] + lik_los[bins == 4] + 0)
summary(model_lm3)

df = data.frame(prior = (quantile(abs(prior_los))[2:5] + quantile(abs(prior_los))[1:4])/2.0,  
                coeff = c(model_lm0$coefficients[1], model_lm1$coefficients[1], model_lm2$coefficients[1], model_lm3$coefficients[1]))
plot2 <- ggplot(df, aes(y = coeff, x = prior)) +
  geom_point() + geom_line() + ggtitle("Prior binned regression coeff") 
plot2

# All plots together
grid.arrange(plot1, plot2, nrow = 2)
g <- arrangeGrob(plot1, plot2, nrow = 2)
ggsave(file = "Intuition.pdf", g)

