library(tidyverse)
library(ggplot2)
all <-  read.csv("results.csv")


###########################################
#   View mean +/- sd for a given variant  #
###########################################
view_variant <- function(num_added, template_dim, adaptivity) {
  dat <- all %>% filter(addedCompW == num_added & nmf_type==template_dim & fixW==adaptivity)
  dat <- subset(dat, select = c(noise.lvl, F))
  dat <- dat %>% group_by(noise.lvl) %>% summarise_all(list(F.mean=mean, F.sd=sd))
  sorted <- dat[order(dat$noise.lvl),]
  View(sorted)
}
view_variant(0, "NMFD", "adaptive")


#####################################################
#   Plot performance for adaptive and fixed NMF(D)  #
#####################################################
dat <- all %>% filter(addedCompW==0 & (beta==Inf | beta==0))
dat$type <- paste(dat$fixW, dat$nmf_type)
dat <- subset(dat, select = c(type, noise.lvl, F))
dat <- dat %>% group_by(type, noise.lvl) %>% summarise_all(list(F.mean=mean, F.sd=sd))
ggplot(dat, mapping = aes(x = noise.lvl, y = F.mean, color = type)) + geom_line() + xlab("Noise level") + ylab("F-measure") + labs(color = "") + 
  geom_errorbar(aes(ymin=F.mean-F.sd, ymax=F.mean+F.sd), width=.1,
                position=position_dodge(.1))


#################################################################################
#   Plot performance for adaptive, fixed, and semi-adaptive NMF(D) with beta=4  #
#################################################################################
dat <- all %>% filter(addedCompW==0 & (beta==Inf | beta==0 | beta==4))
dat <- subset(dat, select = -c(P, R, addedCompW, noise, Sample, fixW))
dat <- dat %>% group_by(beta, noise.lvl, nmf_type) %>% summarise_all(list(F.mean=mean, F.sd=sd))
#View(dat)
dat$beta[dat$beta==0] <- "adaptive"
dat$beta[dat$beta==Inf] <- "fixed"
dat$beta[dat$beta==4] <- "semi"
ggplot(dat, mapping = aes(x = noise.lvl, y = F.mean, color = as.factor(beta))) + geom_line() + xlab("Noise level") + ylab("F-measure") + labs(color = "Adaptivity") + 
  geom_errorbar(aes(ymin=F.mean-F.sd, ymax=F.mean+F.sd), width=.1,
                position=position_dodge(.1)) +
  facet_grid(vars(nmf_type))


#########################################################
#   Plot NMFD performance on all adaptivity conditions  #
#########################################################

# with error bars
dat <- all %>% filter(addedCompW==0 & nmf_type=="NMFD")
dat <- subset(dat, select = -c(P, R, addedCompW, noise, Sample, fixW))
dat <- dat %>% group_by(beta, noise.lvl, nmf_type) %>% summarise_all(list(F.mean=mean, F.sd=sd))
#View(dat)
dat$beta[dat$beta==0] <- "adaptive"
dat$beta[dat$beta==Inf] <- "fixed"
ggplot(dat, mapping = aes(x = noise.lvl, y = F.mean, color = as.factor(beta))) + geom_line() + xlab("Noise level") + ylab("F-measure") + labs(color = "beta") + 
  geom_errorbar(aes(ymin=F.mean-F.sd, ymax=F.mean+F.sd), width=.1,
                position=position_dodge(.1)) +
  facet_grid(vars(nmf_type)) +
  ylim(0,1)

# without error bars
dat <- all %>% filter(addedCompW==0 & nmf_type=="NMFD")
dat <- subset(dat, select = -c(P, R, addedCompW, noise, Sample, fixW))
dat <- dat %>% group_by(beta, noise.lvl, nmf_type) %>% summarise_all(list(F.mean=mean, F.sd=sd))
#View(dat)
dat$beta[dat$beta==0] <- "adaptive"
dat$beta[dat$beta==Inf] <- "fixed"
ggplot(dat, mapping = aes(x = noise.lvl, y = F.mean, color = as.factor(beta))) + geom_line() + xlab("Noise level") + ylab("F-measure") + labs(color = "beta") +
  facet_grid(vars(nmf_type)) +
  ylim(0,1)


########################################################
#   Plot NMF performance on all adaptivity conditions  #
########################################################

# with error bars
dat <- all %>% filter(addedCompW==0 & nmf_type=="NMF")
dat <- subset(dat, select = -c(P, R, addedCompW, noise, Sample, fixW))
dat <- dat %>% group_by(beta, noise.lvl, nmf_type) %>% summarise_all(list(F.mean=mean, F.sd=sd))
#View(dat)
dat$beta[dat$beta==0] <- "adaptive"
dat$beta[dat$beta==Inf] <- "fixed"
ggplot(dat, mapping = aes(x = noise.lvl, y = F.mean, color = as.factor(beta))) + geom_line() + xlab("Noise level") + ylab("F-measure") + labs(color = "beta") + 
  geom_errorbar(aes(ymin=F.mean-F.sd, ymax=F.mean+F.sd), width=.1,
                position=position_dodge(.1)) +
  facet_grid(vars(nmf_type)) +
  ylim(0,1)

# without error bars
dat <- all %>% filter(addedCompW==0 & nmf_type=="NMF")
dat <- subset(dat, select = -c(P, R, addedCompW, noise, Sample, fixW))
dat <- dat %>% group_by(beta, noise.lvl, nmf_type) %>% summarise_all(list(F.mean=mean, F.sd=sd))
#View(dat)
dat$beta[dat$beta==0] <- "adaptive"
dat$beta[dat$beta==Inf] <- "fixed"
ggplot(dat, mapping = aes(x = noise.lvl, y = F.mean, color = as.factor(beta))) + geom_line() + xlab("Noise level") + ylab("F-measure") + labs(color = "beta") +
  facet_grid(vars(nmf_type)) +
  ylim(0,1)


################
#  Figure 3.1  #
################
dat <- all %>% filter(addedCompW==0 & ((beta==Inf & nmf_type=="NMF") | (beta==0 & nmf_type=="NMFD")))
dat$type <- paste(dat$fixW, dat$nmf_type)
dat <- subset(dat, select = c(type, noise.lvl, F))
dat <- dat %>% group_by(type, noise.lvl) %>% summarise_all(list(F.mean=mean, F.sd=sd))
ggplot(dat, mapping = aes(x = noise.lvl, y = F.mean, color = type)) + geom_line() + xlab("Noise level") + ylab("F-measure") + labs(color = "") + 
  geom_errorbar(aes(ymin=F.mean-F.sd, ymax=F.mean+F.sd), width=.1,
                position=position_dodge(.1)) +
  ylim(0,1)


################
#  Figure 3.2  #
################
plot_semi <- function(dat) {
  dat <- dat %>% filter(fixW!="fixed" & addedCompW==0)
  dat <- subset(dat, select = c(nmf_type, beta, noise.lvl, F))
  dat <- dat %>% group_by(nmf_type, beta, noise.lvl) %>% summarise_all(mean)
  ggplot(dat, mapping = aes(x = noise.lvl, y = beta, fill = F)) +
    geom_tile() +
    facet_grid(vars(nmf_type)) +
    xlab("Noise level")
}
plot_semi(all)


################
#  Figure 3.3  #
################
plot_addedComp <- function(dat) {
  dat <- dat %>% filter((fixW=="fixed" | fixW=="adaptive") & noise.lvl != 0)
  dat$type <- paste(dat$fixW, dat$nmf_type)
  dat <- subset(dat, select = -c(Sample, noise, fixW, nmf_type, P, R, beta))
  dat <- dat %>% group_by(type, addedCompW, noise.lvl) %>% summarise_all(list(F.mean=mean, F.sd=sd))
  ggplot(dat, mapping = aes(x = addedCompW, y = F.mean, color=type)) + geom_line() +
    facet_grid(vars(noise.lvl)) +
    geom_errorbar(aes(ymin=F.mean-F.sd, ymax=F.mean+F.sd), width=.1,
                  position=position_dodge(.15)) +
    labs(color = "") +
    xlab("# added components") +
    ylab("F-measure")
}
plot_addedComp(all)


################
#  Figure 3.4  #
################
plot_semi <- function(dat) {
  dat <- dat %>% filter(fixW=="semi" & noise.lvl != 0)
  dat <- subset(dat, select = c(nmf_type, beta, addedCompW, noise.lvl, F))
  dat <- dat %>% group_by(nmf_type, beta, addedCompW, noise.lvl) %>% summarise_all(mean)
  ggplot(dat, mapping = aes(x = addedCompW, y = beta, fill = F)) +
    geom_tile() +
    facet_grid(vars(nmf_type), vars(noise.lvl)) +
    xlab("# added components")
}
plot_semi(all)


#########################
#   Tables D.1 and D.2  #
#########################
view_table <- function(type, all) {
  dat <- all %>% filter(addedCompW==0 & nmf_type==type)
  dat <- subset(dat, select = c(beta, noise.lvl, F))
  dat <- dat %>% group_by(beta, noise.lvl) %>% summarise_all(list(F.mean=mean, F.sd=sd))
  nrow(dat)
  return(dat)
}
dat <- view_table("NMF", all)
dat <- view_table("NMFD", all)
View(dat)

beta = 0
for (i in 1:nrow(dat)) {
  cat("$", round(dat[i,]$F.mean, digits = 3), "\\pm", round(dat[i,]$F.sd, digits = 3), "$")
  if(dat[i+1,]$beta == beta) {
    cat(" & ")
  } else {
    cat(" \\\\ \n")
    beta <- dat[i+1,]$beta
    if(beta == Inf){
      cat("Fixed & ")
    } else {
      cat("Semi $\\beta=", beta ,"$", " & ")
    }
  }
}


