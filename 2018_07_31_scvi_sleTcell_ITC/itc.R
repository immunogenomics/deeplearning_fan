
setwd("Documents/GitHub/deeplearning_fan/2018_07_31_scvi_sleTcell_ITC/")
library(gdata)
library(ggplot2)
library(RColorBrewer)
library(viridis)
library(plyr)
library(patchwork)
library(data.table)

ITC <- load("../data/ITC_rnaseq_qc_norm_coreCellT.rda")
ens <- rownames(log2tpm)
dim(log2tpm)

ens_to_gene <- read.xls("../../../HMS/text_mining_single_cell_Gaurav/Gene_Gene_rank_matrix_test/gene_ID_name_ensemble_gencodeV24.xls")
rownames(ens_to_gene) <- ens_to_gene$ENSEMBLE_ID

merge_data <- merge(ens_to_gene, log2tpm, by = "row.names")
dim(merge_data)
merge_data[1:4,]

merge_data <- merge_data[, -c(1,2,4)]
merge_data$GENE_NAME <- as.character(merge_data$GENE_NAME)
rownames(merge_data) <- make.names(merge_data$GENE_NAME, unique=TRUE)

merge_data <- merge_data[, -1]
dim(merge_data)

all(colnames(merge_data) == m$sampleID)
table(m$cell_type)


# SLE data
library(stringr)
library(data.table)

sle_meta <- fread("gunzip -c ../data/SLE/cell_metadata.txt.gz")
sle_meta <- as.data.frame(sle_meta)
sle_log2cpm <- fread("gunzip -c ../data/SLE/gene_by_cell_exp_mat.txt.gz")
sle_log2cpm <- as.data.frame(sle_log2cpm)
rownames(sle_log2cpm) <- sle_log2cpm$V1
sle_log2cpm <- sle_log2cpm[, -1]
dim(sle_log2cpm)
all(sle_meta$name == colnames(sle_log2cpm))

sle_log2cpm <- sle_log2cpm[, grep("CT", sle_meta$cluster)]
sle_meta <- sle_meta[grep("CT", sle_meta$cluster), ]
all(sle_meta$sample == colnames(sle_log2cpm))


# Intersected genes
ITC_inter <- merge_data[which(rownames(merge_data) %in% intersect(rownames(merge_data), rownames(sle_log2cpm))),]
dim(ITC_inter)
SLE_inter <- sle_log2cpm[which(rownames(sle_log2cpm) %in% intersect(rownames(merge_data), rownames(sle_log2cpm))),]
dim(ITC_inter)

# ITC_inter <- ITC_inter[ order(match(rownames(ITC_inter), rownames(ITC_inter))), ]
ITC_inter <- ITC_inter[order(rownames(ITC_inter)),]
all(rownames(ITC_inter) == rownames(SLE_inter))


# Merge together
ITC_scale <- apply(as.matrix(ITC_inter), MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
dim(ITC_scale)
SLE_scale <- apply(as.matrix(SLE_inter), MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
dim(SLE_scale)

ITC_SLE_exp <- cbind.data.frame(ITC_scale, SLE_scale)
dim(ITC_SLE_exp)



all(colnames(ITC_scale) == m$sampleID)
all(sle_meta$name == colnames(SLE_scale))
ITC_SLE_meta = data.frame(
  sample = c(m$sampleID, sle_meta$name),
  cluster = c(m$cell_type, sle_meta$cluster),
  data_type = c(rep("ITC", nrow(m)), rep("SLE", nrow(sle_meta)))
)
dim(ITC_SLE_meta)
saveRDS(ITC_SLE_meta, "../data/ITC_SLE_meta.rds")

save_tsv <- function(x, filename) {
  data.table::fwrite(x, filename, sep = "\t")
  system(sprintf("pigz --force --best %s", filename))
}

ITC_SLE_meta$data_type <- as.character(ITC_SLE_meta$data_type)
ITC_SLE_meta$data_type[which(ITC_SLE_meta$data_type == "ITC")] <- 0
ITC_SLE_meta$data_type[which(ITC_SLE_meta$data_type == "SLE")] <- 1
ITC_SLE_meta$cluster_num <- as.numeric(ITC_SLE_meta$cluster)
all(ITC_SLE_meta$sample == colnames(ITC_SLE_exp))

save_tsv(ITC_SLE_meta, "../2018_07_31_scvi_sleTcell_ITC/ITC_SLE_meta.tsv")

z <- gzfile("../2018_07_31_scvi_sleTcell_ITC/ITC_SLE_exp.csv.gz")
write.csv(ITC_SLE_exp, z)


# Plot the results scVI
latent <- read.csv("my_latent.csv", header = FALSE, sep = ",", quote = "\"")
rownames(latent) <- colnames(ITC_SLE_exp)
latent$cluster <- ITC_SLE_meta$cluster
latent$data_type <- ITC_SLE_meta$data_type

ggplot() +
  geom_point(
    data = latent[sample(nrow(latent)),],
    mapping = aes_string(x = "V1", y = "V2", fill = "cluster"),
    size = 2, stroke = 0.05, shape = 21
  ) +
  # scale_fill_manual(values = meta_colors$cell_type, name = "") +
  # scale_fill_manual(values = meta_colors$fine_cluster, name = "") +
  # scale_fill_manual(values = meta_colors$data_type, name = "") +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_bw(base_size = 10) +
  theme(
    axis.text = element_blank(), 
    axis.ticks = element_blank(), 
    panel.grid = element_blank()
  ) 


