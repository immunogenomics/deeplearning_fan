
setwd("Documents/GitHub/deeplearning/2018_07_17_test_scvis/")
library(ggplot2)
library(RColorBrewer)
source("../../amp_phase1_ra/R/meta_colors.R")

# Plot scvis results of integrating RA and SLE datasets

# Plot RA embeddings
emb_ra <- read.table(file = 'output_train_ra_normalization/ra/perplexity_10_regularizer_0.001_batch_size_512_learning_rate_0.01_latent_dimension_2_activation_ELU_seed_1_iter_3000.tsv', sep = '\t', header = TRUE)
dim(emb_ra)
raw_meta <- readRDS("../../amp_phase1_ra_viewer/data/celseq_synovium_meta_5265cells_paper.rds")

cluster <- read.table(file = '../data/amp_ra_cluster_celllabel.tsv', sep = '\t', header = TRUE)
emb_ra$cluster <- cluster$amp_ra_cluster_label
emb_ra$general_cell_type <- raw_meta$cell_type
emb_ra$condition <- raw_meta$disease
emb_ra$plate <- raw_meta$plate
emb_ra$patient <- raw_meta$sample


ggplot() +
  geom_point(
    data = emb_ra,
    mapping = aes_string(x = "z_coordinate_0", y = "z_coordinate_1", fill = "cluster"),
    size = 1.5, stroke = 0.05, shape = 21
  ) +
  scale_fill_manual(values = meta_colors$fine_cluster, name = "") +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_bw(base_size = 9) +
  theme(
      axis.text = element_blank(), 
      axis.ticks = element_blank(), 
      panel.grid = element_blank()
  ) 
ggsave(file = paste("scvis_ra_sample", 
                    "perplexity_10_regularizer_0.001_batch_size_512_learning_rate_0.01_latent_dimension_2_activation_ELU_seed_1_iter_3000", 
                    ".png", sep = ""), width = 7, height = 4.5, dpi = 300)
dev.off()


# -------------------
# Plot SLE embeddings

meta_colors$sle_cluster <- c(
  "CB0" = "#E41A1C",
  "CB1" = "#377EB8",
  "CB2a" = "#4DAF4A",
  "CB2b" = "#984EA3",
  "CB3" = "#FF7F00",
  "CD0" = "#FFFF33",
  "CE0" = "#A65628",
  "CM0" = "#F781BF",
  "CM1" = "#999999",
  "CM2" = "#66C2A5",
  "CM3" = "#FC8D62",
  "CM4" = "#8DA0CB",
  "CT0a" = "#E78AC3",
  "CT0b" = "#A6D854",
  "CT1" = "#FFD92F",
  "CT2" = "#E5C494",
  "CT3a" = "#B3B3B3",
  "CT3b" = "#8DD3C7",
  "CT4" = "#FFFFB3",
  "CT5a" = "#BEBADA",
  "CT5b" = "#FB8072",
  "CT6" = "#80B1D3"
)

meta_colors$sle_type <- c(
  "B cells" = "#E31A1C",
  "Dividing cells" =  "#E5C494",
  "T/NK cells" = "#A65628", 
  "Epithelial cells"  = "dimgrey",
  "Myeloid cells"  = "#DE77AE" 
)
meta_colors$data_type <- c(
  "RA" = "#FC8D62",
  "SLE" =  "#66C2A5"
)
meta_colors$general_cell_type <- c(
  "B cell"  =   "#E31A1C",    
  "Empty"   =  "grey",    
  "Fibroblast"  =  "#1F78B4",
  "Monocyte"  =  "#FF7F00", 
  "T cell" =  "#B15928",
  
  "Epithelial cells" = "#A6CEE3",
  "Dividing cells"  = "#B2DF8A",
  "Myeloid cells"  = "#FDBF6F", 
  "T/NK cells"  = "#FFFF99",
  "B cells" ="#FB9A99" 
)

emb_sle <- read.table(file = '2018_07_17_test_scvis/output/sle/perplexity_10_regularizer_0.001_batch_size_512_learning_rate_0.01_latent_dimension_2_activation_ELU_seed_1_map.tsv', sep = '\t', header = TRUE)
dim(emb_sle)

cluster <- read.table(file = 'data/amp_sle_cluster_celllabel.tsv', sep = '\t', header = TRUE)
emb_sle$cluster <- cluster$amp_sle_cluster_label

raw_meta <- fread("gunzip -c data/SLE/cell_metadata.txt.gz")
raw_meta <- as.data.frame(raw_meta)
emb_sle$general_cell_type <- raw_meta$general_cell_type
emb_sle$condition <- raw_meta$condition
emb_sle$plate <- raw_meta$plate
emb_sle$patient <- raw_meta$patient

ggplot() +
  geom_point(
    data = emb_sle,
    mapping = aes_string(x = "z_coordinate_0", y = "z_coordinate_1", fill = "general_cell_type"),
    size = 2, stroke = 0.05, shape = 21
  ) +
  scale_fill_manual(values = meta_colors$sle_type, name = "") +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_bw(base_size = 20) +
  theme(
    axis.text = element_blank(), 
    axis.ticks = element_blank(), 
    panel.grid = element_blank()
  ) 


# Putting RA and SLE together
emb_ra$data_type <- rep("RA", nrow(emb_ra))
emb_sle$data_type <- rep("SLE", nrow(emb_sle))
both <- rbind.data.frame(emb_ra, emb_sle)

ggplot() +
  geom_point(
    data = both[sample(nrow(both)),],
    mapping = aes_string(x = "z_coordinate_0", y = "z_coordinate_1", fill = "general_cell_type"),
    # mapping = aes_string(x = "z_coordinate_0", y = "z_coordinate_1", fill = "data_type"),
    size = 1.5, stroke = 0.05, shape = 21
  ) +
  scale_fill_manual(values = meta_colors$general_cell_type, name = "") +
  # scale_fill_manual(values = meta_colors$data_type, name = "") +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_bw(base_size = 15) +
  theme(
    axis.text = element_blank(), 
    axis.ticks = element_blank(), 
    panel.grid = element_blank()
  ) 
ggsave(file = paste("scvis_ra_sle_generalcelltype", 
                    ".png", sep = ""), width = 7, height = 4.5, dpi = 300)
dev.off()




