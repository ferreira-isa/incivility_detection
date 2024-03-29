library(ggplot2) 
library(hrbrthemes)
library(plotly)
library(viridis)
library(ggh4x)
library(scales)

IMG_DIR=("plots/")

# Heatmap rq1
W =  10
H =  10
data <- read.csv(file="input/rq1_civil_uncivil.csv", sep = ",")
data <- data[data$correctly_predicted. == "FALSE",]
#data <- data[data$X != "0.00%",]
data

data$percentage <- as.double(gsub("%", "", data$X.))

data$quotation_tbdf <- as.character(data$quotation_tbdf)

cols <-function(n) {
  colorRampPalette(c("forestgreen", "gold1", "orange", " tomato3" , " red4","red2"))(5)                          }

p <- ggplot(data, aes(x = model, y = reorder(quotation_tbdf, -order), fill = percentage)) +
  geom_tile(alpha=0.9) +
  theme_bw() + 
  facet_grid(label_correct~data, scales='free', space="free")+
  xlab("")+
  ylab("TBDFs") +
  theme(axis.text.x=element_text(colour="black")) +
  theme(axis.text.y=element_text(colour="black")) +
  theme(axis.text = element_text(size = 12))      +
  theme(legend.text = element_text(size = 12)) +
  theme(axis.text.x = element_text(size = 12)) +
  theme(axis.title = element_text(size = 12)) +
  theme(axis.title.y = element_text(size = 12)) +
  theme(panel.spacing = unit(0.2, "cm")) +
  theme(strip.text.x = element_text(size = 12), strip.text.y = element_text(size = 12), legend.text=element_text(size=12), legend.title=element_text(size=12)) +
  scale_fill_gradientn(colours = cols(length(mycols)), breaks=c(0, 25, 50, 75, 100), labels=c(0, 25, 50, 75, 100), limits=c(0, 100)) +
  guides(fill = guide_colorbar(title ="% misclassified sentences per TBDF",
                               label.position = "bottom",
                               title.position = "left", title.vjust = 0.75,
                               frame.colour = "black",
                               barwidth = 10,
                               barheight = 1.5)) +
  geom_text(aes(label = X.), color = "black", size = 3.5)+ theme(legend.position="bottom")
p
ggsave(paste(IMG_DIR,'heatmap_discussion_rq1_civil_uncivil.pdf', sep=""), plot = p, width = W, height = H, units = c("in"))
extrafont::embed_fonts(paste(IMG_DIR,'heatmap_discussion_rq1_civil_uncivil.pdf', sep=""), outfile=paste(IMG_DIR,'heatmap_discussion_rq1_civil_uncivil.pdf', sep=""))
ggsave(paste(IMG_DIR,'heatmap_discussion_rq1_civil_uncivil.pdf', sep=""), plot = p, width = W, height = H, units = c("in"))



 # Heatmap rq2
W =  10
H =  10
data <- read.csv(file="input/rq2_civil_uncivil.csv", sep = ",")
data <- data[data$correctly_predicted. == "FALSE",]
#data <- data[data$X != "0.00%",]
data

data$percentage <- as.double(gsub("%", "", data$X.))

data$quotation_tbdf <- as.character(data$quotation_tbdf)

cols <-function(n) {
  colorRampPalette(c("forestgreen", "gold1", "orange", " tomato3" , " red4","red2"))(5)                          }

p <- ggplot(data, aes(x = model, y = reorder(quotation_tbdf, -order), fill = percentage)) +
  geom_tile(alpha=0.9) +
  theme_bw() + 
  facet_grid(label_correct~data, scales='free', space="free")+
  xlab("")+
  ylab("TBDFs") +
  theme(axis.text.x=element_text(colour="black")) +
  theme(axis.text.y=element_text(colour="black")) +
  theme(axis.text = element_text(size = 12))      +
  theme(legend.text = element_text(size = 12)) +
  theme(axis.text.x = element_text(size = 12)) +
  theme(axis.title = element_text(size = 12)) +
  theme(axis.title.y = element_text(size = 12)) +
  theme(panel.spacing = unit(0.2, "cm")) +
  guides(fill = guide_colorbar(title ="% misclassified sentences per TBDF",
                               label.position = "bottom",
                               title.position = "left", title.vjust = 0.75,
                               frame.colour = "black",
                               barwidth = 10,
                               barheight = 1.5)) +
  theme(strip.text.x = element_text(size = 12), strip.text.y = element_text(size = 12), legend.text=element_text(size=12), legend.title=element_text(size=12)) +
  scale_fill_gradientn(colours = cols(length(mycols)), breaks=c(0, 25, 50, 75, 100), labels=c(0, 25, 50, 75, 100), limits=c(0, 100)) +
  geom_text(aes(label = X.), color = "black", size = 3.5)+ theme(legend.position="bottom")
p
ggsave(paste(IMG_DIR,'heatmap_discussion_rq2_civil_uncivil.pdf', sep=""), plot = p, width = W, height = H, units = c("in"))
extrafont::embed_fonts(paste(IMG_DIR,'heatmap_discussion_rq2_civil_uncivil.pdf', sep=""), outfile=paste(IMG_DIR,'heatmap_discussion_rq2_civil_uncivil.pdf', sep=""))
ggsave(paste(IMG_DIR,'heatmap_discussion_rq2_civil_uncivil.pdf', sep=""), plot = p, width = W, height = H, units = c("in"))

# Heatmap rq3
W =  10
H =  10
data <- read.csv(file="input/rq3_civil_uncivil.csv", sep = ",")
data <- data[data$correctly_predicted. == "FALSE",]
#data <- data[data$X != "0.00%",]
data

data$percentage <- as.double(gsub("%", "", data$X.))

data$quotation_tbdf <- as.character(data$quotation_tbdf)

cols <-function(n) {
  colorRampPalette(c("forestgreen", "gold1", "orange", " tomato3" , " red4","red2"))(5)                          }

p <- ggplot(data, aes(x = model, y = reorder(quotation_tbdf, -order), fill = percentage)) +
  geom_tile(alpha=0.9) +
  theme_bw() + 
  facet_grid(label_correct~data, scales='free', space="free")+
  xlab("")+
  ylab("TBDFs") +
  theme(axis.text.x=element_text(colour="black")) +
  theme(axis.text.y=element_text(colour="black")) +
  theme(axis.text = element_text(size = 12))      +
  theme(legend.text = element_text(size = 12)) +
  theme(axis.text.x = element_text(size = 12)) +
  theme(axis.title = element_text(size = 12)) +
  theme(axis.title.y = element_text(size = 12)) +
  theme(panel.spacing = unit(0.2, "cm")) +
  guides(fill = guide_colorbar(title ="% misclassified sentences per TBDF",
                               label.position = "bottom",
                               title.position = "left", title.vjust = 0.75,
                               frame.colour = "black",
                               barwidth = 10,
                               barheight = 1.5)) +
  theme(strip.text.x = element_text(size = 12), strip.text.y = element_text(size = 12), legend.text=element_text(size=12), legend.title=element_text(size=12)) +
  scale_fill_gradientn(colours = cols(length(mycols)), breaks=c(0, 25, 50, 75, 100), labels=c(0, 25, 50, 75, 100), limits=c(0, 100)) +
  geom_text(aes(label = X.), color = "black", size = 3.5)+ theme(legend.position="bottom")
p
ggsave(paste(IMG_DIR,'heatmap_discussion_rq3_civil_uncivil.pdf', sep=""), plot = p, width = W, height = H, units = c("in"))
extrafont::embed_fonts(paste(IMG_DIR,'heatmap_discussion_rq3_civil_uncivil.pdf', sep=""), outfile=paste(IMG_DIR,'heatmap_discussion_rq3_civil_uncivil.pdf', sep=""))
ggsave(paste(IMG_DIR,'heatmap_discussion_rq3_civil_uncivil.pdf', sep=""), plot = p, width = W, height = H, units = c("in"))
