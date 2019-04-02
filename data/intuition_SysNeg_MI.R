library(jsonlite)
library(ggplot2)
require(gridExtra)

Nbins = 18

setwd("~/GitHub/LearningToInfer/data")

fn = "N_part19__expt_nameSN__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_blocks400__train_epoch100__train_lr0.05__plot_data"
data <- fromJSON(txt=fn)


subsample_unif <- function(samples, bins){
  h = hist(samples, bins)
  thresh = min(h$density)
  bins = .bincode(samples, breaks = h$breaks)
  accept_prob = 1.0 - (h$density[bins] - thresh)/h$density[bins]
  return (lapply(accept_prob, function(p) rbinom(1, 1, p)) > 0.5)
}

true0 <- data$hrms
model0 <- data$ams

sub_index = subsample_unif(true0, Nbins)
true <- true0[sub_index]
model <- model0[sub_index]

df = data.frame(x = c(true, true),
                y = c(true, model),
                kind = c(rep("True", length(true)), rep("model", length(model))))


df = data.frame(x = true,
                y = model)
df$cut_true <- cut(df$x, breaks = seq(-0, 1, 0.025))

p <- ggplot(df,  aes(y = y, x = factor(cut_true))) +
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  geom_point(size = 0.001)+
  ylab("Approximate posterior") + 
  xlab("True posterior") + 
  ylim(c(0, 1)) +
  theme_classic() +
  theme(axis.text.x = element_blank(), text = element_text(size=18)) 
p0<-ggExtra::ggMarginal(p, type = "histogram", bins = Nbins, margins = 'y', size = 2.5)
p0
ggsave("Demo_low.pdf", p0)

# *****************

fn = "N_part19__expt_nameSN__NHID5__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_blocks400__train_epoch500__train_lr0.05__plot_data"
data <- fromJSON(txt=fn)

true0 <- data$hrms
model0 <- data$ams

sub_index = subsample_unif(true0, Nbins)
true <- true0[sub_index]
model <- model0[sub_index]

df = data.frame(x = c(true, true),
                y = c(true, model),
                kind = c(rep("True", length(true)), rep("model", length(model))))


df = data.frame(x = true,
                y = model)
df$cut_true <- cut(df$x, breaks = seq(-0, 1, 0.025))

p <- ggplot(df,  aes(y = y, x = factor(cut_true))) +
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  geom_point(size = 0.001)+
  ylab("Approximate posterior") + 
  xlab("True posterior") + 
  ylim(c(0, 1)) +
  theme_classic() +
  theme(axis.text.x = element_blank(), text = element_text(size=18)) 
p0<-ggExtra::ggMarginal(p, type = "histogram", bins = Nbins, margins = 'y', size = 2.5)
p0
ggsave("Demo_high.pdf", p0)


#*************************


fn = "N_part19__expt_nameSN__NHID2__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_blocks400__train_epoch500__train_lr0.05__plot_data"
data <- fromJSON(txt=fn)


subsample_unif <- function(samples, bins){
  h = hist(samples, bins)
  thresh = min(h$density)
  bins = .bincode(samples, breaks = h$breaks)
  accept_prob = 1.0 - (h$density[bins] - thresh)/h$density[bins]
  return (lapply(accept_prob, function(p) rbinom(1, 1, p)) > 0.5)
}

true0 <- data$hrms
model0 <- data$ams

sub_index = subsample_unif(true0, Nbins)
true <- true0[sub_index]
model <- model0[sub_index]

df = data.frame(x = c(true, true),
                y = c(true, model),
                kind = c(rep("True", length(true)), rep("model", length(model))))


df = data.frame(x = true,
                y = model)
df$cut_true <- cut(df$x, breaks = seq(-0, 1, 0.025))

p <- ggplot(df,  aes(y = y, x = factor(cut_true))) +
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  geom_point(size = 0.001)+
  ylab("Approximate posterior") + 
  xlab("True posterior") + 
  ylim(c(0, 1)) +
  theme_classic() +
  theme(axis.text.x = element_blank(), text = element_text(size=18)) 
p0<-ggExtra::ggMarginal(p, type = "histogram", bins = Nbins, margins = 'y', size = 2.5)
p0
ggsave("Demo_medium.pdf", p0)


#*************************


fn = "N_part199__expt_nameSN__NHID1__NONLINrbf__L20.0__test_epoch0__test_lr0.0__train_blocks400__train_epoch0__train_lr0.05__plot_data"
data <- fromJSON(txt=fn)


subsample_unif <- function(samples, bins){
  h = hist(samples, bins)
  thresh = min(h$density)
  bins = .bincode(samples, breaks = h$breaks)
  accept_prob = 1.0 - (h$density[bins] - thresh)/h$density[bins]
  return (lapply(accept_prob, function(p) rbinom(1, 1, p)) > 0.5)
}

true0 <- data$hrms
model0 <- data$ams

sub_index = subsample_unif(true0, Nbins)
true <- true0[sub_index]
model <- model0[sub_index]

df = data.frame(x = c(true, true),
                y = c(true, model),
                kind = c(rep("True", length(true)), rep("model", length(model))))


df = data.frame(x = true,
                y = model)
df$cut_true <- cut(df$x, breaks = seq(-0, 1, 0.025))

p <- ggplot(df,  aes(y = y, x = factor(cut_true))) +
  geom_boxplot(outlier.shape = NA,outlier.colour = NA) +
  geom_point(size = 0.001)+
  ylab("Approximate posterior") + 
  xlab("True posterior") + 
  ylim(c(0, 1)) +
  theme_classic() +
  theme(axis.text.x = element_blank(), text = element_text(size=18)) 
p0<-ggExtra::ggMarginal(p, type = "histogram", bins = Nbins, margins = 'y', size = 2.5)
p0
ggsave("Demo_nolearning.pdf", p0)
