library(jsonlite)
library(ggplot2)
require(gridExtra)
setwd("~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}
# Load data for Philip Edwards expt
# files = list.files(path = ".")
# # file = files[lapply(files, function(strs, pattern) grepl(pattern, strs), pattern = "PM")]
# for (f in files){
#   if (grepl('expt_namePM', f)) {
#     fn <- f
#   }
# }

fn = "N_part49__expt_namePM__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch30__train_lr0.05__plot_data"
PM_data <- fromJSON(txt=fn)
prior_los <- log_odds(PM_data$priors)
true_los <- log_odds(PM_data$hrms)
lik_los <- true_los - prior_los
model_los <- log_odds(PM_data$ams)


# prior_los <- log_odds(c(PM_data$priors, EU_data$ID_priors))
# true_los <- log_odds(c(PM_data$hrms, EU_data$ID_hrms))
# lik_los <- true_los - prior_los
# model_los <- log_odds(c(PM_data$ams, EU_data$ID_ams))


# Bin the lik_los

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
  geom_point() + geom_line() + ggtitle("Lik binned regression coeff") + ylim(c(0.2, 1.2))
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
  geom_point() + geom_line() + ggtitle("Prior binned regression coeff") + ylim(c(0.6, 0.9))
plot2

# All plots together
grid.arrange(plot1, plot2, nrow = 2)
g <- arrangeGrob(plot1, plot2, nrow = 2)
ggsave(file = "PM_log_odds_regression_nhid1.pdf", g)

##########################################

fn = "N_part49__expt_namePM__NHID5__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch30__train_lr0.05__plot_data"
PM_data <- fromJSON(txt=fn)
prior_los <- log_odds(PM_data$priors)
true_los <- log_odds(PM_data$hrms)
lik_los <- true_los - prior_los
model_los <- log_odds(PM_data$ams)


# Bin the lik_los

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
  geom_point() + geom_line() + ggtitle("Lik binned regression coeff") + ylim(c(0.7, 1.3))
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
plot2 <- ggplot(df, aes(y = coeff, x = prior)) + ylim(c(0.7, 1.3)) + 
  geom_point() + geom_line() + ggtitle("Prior binned regression coeff") 
plot2

# All plots together
grid.arrange(plot1, plot2, nrow = 2)
g <- arrangeGrob(plot1, plot2, nrow = 2)
ggsave(file = "PM_log_odds_regression_nhid5.pdf", g)



