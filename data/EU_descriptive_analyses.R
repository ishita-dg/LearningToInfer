library("rjson")
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
EU_data <- fromJSON(file=fn)

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