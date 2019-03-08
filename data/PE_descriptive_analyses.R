library(rjson)
library(ggplot2)
setwd("~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}
# Load data for Philip Edwards expt
# files = list.files(path = ".")
# # file = files[lapply(files, function(strs, pattern) grepl(pattern, strs), pattern = "PE")]
# for (f in files){
#   if (grepl('expt_namePE', f)) {
#     fn <- f
#   }
# }

fn = "N_part19__expt_namePE__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch200__train_lr0.02__plot_data"
PE_data <- fromJSON(file=fn)
prior_los <- log_odds(PE_data$priors)
true_los <- log_odds(PE_data$hrms)
lik_los <- true_los - prior_los
model_los <- log_odds(PE_data$ams)

# By conditions of p value

model_lm <- lm(model_los ~ lik_los + prior_los + 0)
summary(model_lm)

model_lm0 <- lm(model_los[PE_data$conds == 0] ~ lik_los[PE_data$conds == 0] + prior_los[PE_data$conds == 0] + 0)
summary(model_lm0)

model_lm1 <- lm(model_los[PE_data$conds == 1] ~ lik_los[PE_data$conds == 1] + prior_los[PE_data$conds == 1]+ 0)
summary(model_lm1)

model_lm2 <- lm(model_los[PE_data$conds == 2] ~ lik_los[PE_data$conds == 2] + prior_los[PE_data$conds == 2]+ 0)
summary(model_lm2)

df = data.frame(lik = c(0.85, 0.70, 0.55), 
                coeff = c(model_lm0$coefficients[1], model_lm1$coefficients[1], model_lm2$coefficients[1]))
plot <- ggplot(df, aes(y = coeff, x = lik)) +
  geom_point() + geom_line()


# Better to instead bin the lik_los

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
  geom_point() + geom_line() + ggtitle("Binned regression coeff") + ylim(c(0.5, 2.5))
plot1

df = data.frame(posterior = c(true_los, model_los), 
                lik = c(lik_los, lik_los),
                kind = c(rep("True", length(true_los)), rep("model", length(model_los))))
df$cut_lik <- cut(df$lik, breaks = seq(-4.5, 4.5, 0.4))
plot2 <- ggplot(df, aes(y = posterior, x = factor(cut_lik), fill = kind)) +
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  ylim(c(-3, 3))  + ggtitle("Regression") +
  theme(axis.text.x = element_blank())
plot2

# All plots together
grid.arrange(plot1, plot2, nrow = 2)
g<- arrangeGrob(plot1, plot2, nrow = 2)
ggsave(file = "PE_log_odds_regression_nhid1.pdf", g)

####################################
# Changing nhid value
####################################

fn = "N_part19__expt_namePE__NHID5__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch200__train_lr0.02__plot_data"
PE_data <- fromJSON(file=fn)
prior_los <- log_odds(PE_data$priors)
true_los <- log_odds(PE_data$hrms)
lik_los <- true_los - prior_los
model_los <- log_odds(PE_data$ams)

# By conditions of p value

model_lm <- lm(model_los ~ lik_los + prior_los + 0)
summary(model_lm)

model_lm0 <- lm(model_los[PE_data$conds == 0] ~ lik_los[PE_data$conds == 0] + prior_los[PE_data$conds == 0] + 0)
summary(model_lm0)

model_lm1 <- lm(model_los[PE_data$conds == 1] ~ lik_los[PE_data$conds == 1] + prior_los[PE_data$conds == 1]+ 0)
summary(model_lm1)

model_lm2 <- lm(model_los[PE_data$conds == 2] ~ lik_los[PE_data$conds == 2] + prior_los[PE_data$conds == 2]+ 0)
summary(model_lm2)

df = data.frame(lik = c(0.85, 0.70, 0.55), 
                coeff = c(model_lm0$coefficients[1], model_lm1$coefficients[1], model_lm2$coefficients[1]))
plot <- ggplot(df, aes(y = coeff, x = lik)) +
  geom_point() + geom_line()


# Better to instead bin the lik_los

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
  geom_point() + geom_line() + ggtitle("Binned regression coeff") + ylim(c(0.5, 2.5))
plot1

df = data.frame(posterior = c(true_los, model_los), 
                lik = c(lik_los, lik_los),
                kind = c(rep("True", length(true_los)), rep("model", length(model_los))))
df$cut_lik <- cut(df$lik, breaks = seq(-4.5, 4.5, 0.4))
plot2 <- ggplot(df, aes(y = posterior, x = factor(cut_lik), fill = kind)) +
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  ylim(c(-3, 3))  + ggtitle("Regression") +
  theme(axis.text.x = element_blank())
plot2

# All plots together
grid.arrange(plot1, plot2, nrow = 2)
g<- arrangeGrob(plot1, plot2, nrow = 2)
ggsave(file = "PE_log_odds_regression_nhid5.pdf", g)
# 
# # A^B = K
# K = df$lik[1]^df$coeff[1]
# Ks = rep(K, 4)
# Bs = log(Ks)/log(df$lik)

