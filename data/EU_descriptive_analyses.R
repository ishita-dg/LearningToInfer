library(jsonlite)
library(ggplot2)
require(gridExtra)

setwd("~/GitHub/LearningToInfer/data")

log_odds <- function(x){
  return(log(x/(1.0 - x)))
}
# Load data for Eri'c Urn expt
files = list.files(path = ".")
# file = files[lapply(files, function(strs, pattern) grepl(pattern, strs), pattern = "EU")]
for (f in files){
  if (grepl('EU', f)) {
    fn <- f
  }
}
EU_data <- fromJSON(txt=fn)

# Informative data
lik_los <- log_odds(EU_data$ID_liks)
prior_los <- log_odds(EU_data$ID_priors)
true_los <- log_odds(EU_data$ID_hrms)
model_los <- log_odds(EU_data$ID_ams)

true_lm <- lm(true_los ~ lik_los+ prior_los)
summary(true_lm)

ID_model_lm <- lm(model_los ~ lik_los + prior_los)
summary(ID_model_lm)

df = data.frame(posterior = c(true_los, model_los), 
                prior = c(prior_los, prior_los), 
                lik = c(lik_los, lik_los),
                kind = c(rep("True", length(true_los)), rep("model", length(model_los))))

df$cut_prior <- cut(df$prior, breaks = seq(-3, 3, 0.5))
df$cut_lik <- cut(df$lik, breaks = seq(-3, 3, 0.5))


ID_plot_prior <- ggplot(df, aes(y = posterior, x = factor(cut_prior), fill = kind)) +
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  ylim(c(-3, 3))  + ggtitle("Prior coefficient") +
  theme(axis.text.x = element_blank())

ID_plot_lik <- ggplot(df, aes(y = posterior, x = factor(cut_lik), fill = kind)) +
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  ylim(c(-3, 3))  + ggtitle("Likelihood coefficient") +
  theme(axis.text.x = element_blank())

# Uninformative data
lik_los <- log_odds(EU_data$UD_liks)
prior_los <- log_odds(EU_data$UD_priors)
true_los <- log_odds(EU_data$UD_hrms)
model_los <- log_odds(EU_data$UD_ams)

true_lm <- lm(true_los ~ lik_los+ prior_los)
summary(true_lm)

UD_model_lm <- lm(model_los ~ lik_los + prior_los)
summary(UD_model_lm)

df = data.frame(posterior = c(true_los, model_los), 
                prior = c(prior_los, prior_los), 
                lik = c(lik_los, lik_los),
                kind = c(rep("True", length(true_los)), rep("model", length(model_los))))
df$cut_prior <- cut(df$prior, breaks = seq(-3, 3, 0.5))
df$cut_lik <- cut(df$lik, breaks = seq(-3, 3, 0.5))

UD_plot_prior <- ggplot(df, aes(y = posterior, x = factor(cut_prior), fill = kind)) +
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  ylim(c(-3, 3))  + ggtitle("Prior coefficient") +
  theme(axis.text.x = element_blank())

UD_plot_lik <- ggplot(df, aes(y = posterior, x = factor(cut_lik), fill = kind)) +
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  ylim(c(-3, 3))  + ggtitle("Likelihood coefficient") +
  theme(axis.text.x = element_blank())



# All plots together
grid.arrange(ID_plot_lik, ID_plot_prior, UD_plot_lik, UD_plot_prior, layout_matrix = rbind(c(1,2),c(3,4)))
g<- arrangeGrob(ID_plot_lik, ID_plot_prior, UD_plot_lik, UD_plot_prior, layout_matrix = rbind(c(1,2),c(3,4)))
ggsave(file = "EU_log_odds_regression.pdf", g)


### Check for system neglect
all_breaks = c(0.0000000, 0.4054651, 0.6263815, 1.3862944, 2.1972246)

fn = "N_part19__expt_nameEU__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_epoch50__train_lr0.05__plot_data"

data <- fromJSON(txt=fn)
prior_los <- log_odds(data$ID_priors)
true_los <- log_odds(data$ID_hrms)
lik_los <- log_odds(data$ID_liks)
model_los <- log_odds(data$ID_ams)


# Bin the lik_los

bins = .bincode(abs(lik_los), breaks = all_breaks)


model_lm0 <- lm(model_los[bins == 1] ~ prior_los[bins == 1] + lik_los[bins == 1] + 0)
summary(model_lm0)

model_lm1 <- lm(model_los[bins == 2] ~ prior_los[bins == 2] + lik_los[bins == 2] + 0)
summary(model_lm1)

model_lm2 <- lm(model_los[bins == 3] ~ prior_los[bins == 3] + lik_los[bins == 3] + 0)
summary(model_lm2)

model_lm3 <- lm(model_los[bins == 4] ~ prior_los[bins == 4] + lik_los[bins == 4] + 0)
summary(model_lm3)

df = data.frame(prior = (all_breaks[2:5] + all_breaks[1:4])/2.0,  
                coeff = c(model_lm0$coefficients[2], model_lm1$coefficients[2], model_lm2$coefficients[2], model_lm3$coefficients[2]))
plot1 <- ggplot(df, aes(y = coeff, x = prior)) +
  geom_point() + geom_line() + ggtitle("Lik binned regression for ID")
plot1

# Prior bins
data <- fromJSON(txt=fn)
prior_los <- log_odds(data$UD_priors)
true_los <- log_odds(data$UD_hrms)
lik_los <- log_odds(data$UD_liks)
model_los <- log_odds(data$UD_ams)

bins = .bincode(abs(prior_los), breaks = all_breaks)

model_lm0 <- lm(model_los[bins == 1] ~ prior_los[bins == 1] + lik_los[bins == 1] + 0)
summary(model_lm0)

model_lm1 <- lm(model_los[bins == 2] ~ prior_los[bins == 2] + lik_los[bins == 2] + 0)
summary(model_lm1)

model_lm2 <- lm(model_los[bins == 3] ~ prior_los[bins == 3] + lik_los[bins == 3] + 0)
summary(model_lm2)

model_lm3 <- lm(model_los[bins == 4] ~ prior_los[bins == 4] + lik_los[bins == 4] + 0)
summary(model_lm3)

df = data.frame(prior = (all_breaks[2:5] + all_breaks[1:4])/2.0,  
                coeff = c(model_lm0$coefficients[1], model_lm1$coefficients[1], model_lm2$coefficients[1], model_lm3$coefficients[1]))
plot2 <- ggplot(df, aes(y = coeff, x = prior)) +
  geom_point() + geom_line() + ggtitle("Prior binned regression for UD")
plot2

# All plots together
grid.arrange(plot1, plot2, nrow = 2)
g <- arrangeGrob(plot1, plot2, nrow = 2)
ggsave(file = "EU_binned_log_odds_regression.pdf", g)
