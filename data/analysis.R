library(rjson)
library(plyr)
library(ggplot2)
# library(dplyr)

se<-function(x){sd(x)/sqrt(length(x))}
find_AR <- function(r, s, prior, which_urn){
  b_post <- r
  s_post <- s
  # s_post <- s_post*which_urn + (1.0 - s_post)*(1 - which_urn)
  # b_post <- b_post*which_urn + (1.0 - b_post)*(1 - which_urn)
  # prior <- prior*which_urn + (1.0 - prior)*(1 - which_urn)
  
  B_post_odds <- log(b_post/(1.0 - b_post))
  S_post_odds <- log(s_post/(1.0 - s_post))
  BLLR <- B_post_odds - log(prior/(1.0 - prior))
  SLLR <- S_post_odds - log(prior/(1.0 - prior))
  return(SLLR/BLLR)
  
}


pd <- position_dodge(0.06)
cbbPalette <- c("grey50", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")



type = 'test'
dm<-data.frame(Condition=numeric(), frational=numeric(), fpeople=numeric(), prior=numeric())
for (m in c(105:110)){
  modeljson<-fromJSON(file = paste0("nhid2_part",m,"_", type, "preds_disc_epoch30_sg0_f100.1_Nb100_Nt1.json"))
  prior<-c(modeljson$inf_p$p,modeljson$uninf_p$p)
  frational<-c(modeljson$inf_p$hrm,modeljson$uninf_p$hrm)
  fpeople<-c(modeljson$inf_p$am,modeljson$uninf_p$am)
  condition<-rep(c("Informative Prior", "Informative Lik"), each=20)
  dummy<-data.frame(Condition=condition, frational=frational, fpeople=fpeople, prior = prior)
  dm<-rbind(dm, dummy)
}

dm$accuracy_ratio<-find_AR(dm$frational, dm$fpeople, dm$prior)
# dm <- dm[(dm$frational-dm$prior)/(dm$fpeople-dm$prior) > 0,]
dm$rational <- abs(dm$frational - dm$prior)
dm$people <- abs(dm$fpeople - dm$prior)

dm <- dm[dm$rational > 0.00005,]


# bin by prior
N = 10
bins = seq(0, 1, by = 0.1)
dm <- within(dm, group <- as.integer(cut(prior, breaks = bins, include.lowest=TRUE)))

dp<-ddply(dm, ~Condition+group, summarize, mu=mean(accuracy_ratio), se=1.96*se(accuracy_ratio))

dp$group<-mapvalues(dp$group, from=0:N, to = bins)



p8<-ggplot(dp, aes(x=group, y=mu, col=Condition)) +
  geom_point(position =pd)+
  #error bars
  geom_abline(slope = 0.0, intercept = 1.0)+
  geom_errorbar(aes(ymin=mu-se, ymax=mu+se), width=0, size=0.8, position=pd) +
  geom_line(position=pd, size=1) +
  scale_color_manual(values = cbbPalette[c(6,7)])+
  #classic theme, legend on bottom
  theme_classic()+theme(text = element_text(size=12,  family="serif"), 
                        strip.background=element_blank(),
                        legend.position="top")+
  scale_x_continuous(breaks = round(seq(min(0), max(1), by = 0.1),1)) +
  #ylim(c(0,0.35))+
  ylab("Accuracy ratio")+xlab("Prior")+
  #change theme
  theme(text = element_text(size=24, family="sans"))+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.title =  element_blank())+ggtitle("Model AR")
p8
ggsave("Model_AR0.png")


p1<-ggplot(dm, aes(x=rational, y=people, col=Condition)) +
  geom_point(position =pd)+
  scale_color_manual(values = cbbPalette[c(6,7)])+
  #classic theme, legend on bottom
  theme_classic()+theme(text = element_text(size=12,  family="serif"), 
                        strip.background=element_blank(),
                        legend.position="top")+
  scale_x_continuous(breaks = round(seq(min(0), max(1), by = 0.1),1)) +
  #ylim(c(0,0.35))+
  ylab(expression(Delta~"Model"))+xlab(expression(Delta~"Rational"))+
  #change theme
  theme(text = element_text(size=24, family="sans"))+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.title =  element_blank())+ggtitle("Model updates")
# p1

