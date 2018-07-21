
setwd("Documents/GitHub/deeplearning/2018_07_17_test_scvis/")
source("../../amp_phase1_ra/R/pure_functions.R")
library(ggplot2)

# RA data
log2cpm <- readRDS("../../amp_phase1_ra_viewer/data/celseq_synovium_log2_5265cells_paper.rds")
ra_meta <- readRDS("../../amp_phase1_ra_viewer/data/celseq_synovium_meta_5265cells_paper.rds")
dim(log2cpm)

file_mean_sd <- "../data/celseq_synovium_log2cpm_mean_sd.rds"
if (!file.exists(file_mean_sd)) {
  dat_mean_sd <- data.frame(
    mean  = Matrix::rowMeans(log2cpm),
    sd    = apply(log2cpm, 1, sd),
    count = apply(log2cpm, 1, function(x) sum(x > 0))
  )
  xs <- dat_mean_sd$mean
  xs[xs == 0] <- 1
  dat_mean_sd$cv <- dat_mean_sd$sd / xs
  rm(xs)
  dat_mean_sd$density_cv <- with(dat_mean_sd, get_density(mean, cv))
  dat_mean_sd$density_sd <- with(dat_mean_sd, get_density(mean, sd))
  saveRDS(dat_mean_sd, file_mean_sd)
} else {
  dat_mean_sd <- readRDS(file_mean_sd)
}
top_percent <- 0.8
mat_idx <- with(
  dat_mean_sd,
  mean > quantile(mean, top_percent) &
    sd > quantile(sd, top_percent)
)

ra_samples <- log2cpm[mat_idx,]
colnames(ra_meta)[3] <- "Donor.ID"
all(ra_meta$cell_name == colnames(ra_samples))
dim(ra_samples)

# PCA
scale_rows <- function(x) t(scale(t(x)))
pca <- prcomp(~ ., as.data.frame(scale_rows(ra_samples)), scale = TRUE, center = TRUE)
ra_meta <- cbind.data.frame(ra_meta, pca$rotation[, 1:100])

# ggplot() +
#   geom_point(
#     data = ra_meta,
#     mapping = aes_string(x = "PC1", y = "PC2", fill = "cluster"),
#     size = 2, stroke = 0.05, shape = 21
#   )

amp_ra_cluster_label <- ra_meta$cluster
amp_ra_cluster_label <- as.data.frame(amp_ra_cluster_label)
write.table(amp_ra_cluster_label, file = "data/amp_ra_cluster_celllabel.tsv", sep = "\t", row.names = FALSE)

amp_ra_cluster_label$amp_ra_cluster_label <- as.integer(amp_ra_cluster_label$amp_ra_cluster_label)
colnames(amp_ra_cluster_label) <- "Cluster"
write.table(amp_ra_cluster_label, file = "data/amp_ra_cluster_integerlabel.tsv", sep = "\t", row.names = FALSE)

amp_ra_pca100 <- ra_meta[, c(41:140)]
rownames(amp_ra_pca100) <- ra_meta$cell_name
write.table(amp_ra_pca100, file = "data/amp_ra_pca100_cell.tsv", sep = "\t", row.names = FALSE)

rownames(amp_ra_pca100) <- seq(1, nrow(amp_ra_pca100))
write.table(amp_ra_pca100, file = "data/amp_ra_pca100_integer.tsv", sep = "\t", row.names = FALSE)

# Save PCA and cluster label into two files
write.table(amp_ra_pca100, file = "data/amp_ra_pca100.tsv", sep = "\t", row.names = FALSE)
write.table(amp_ra_pca100, file = "data/amp_ra_pca100_cell.tsv", sep = "\t", row.names = FALSE)


# Test and import the saved data files
test1 <- read.table(file = 'data/amp_ra_pca100.tsv', sep = '\t', header = TRUE)
test2 <- read.table(file = 'data/amp_ra_cluster_label.tsv', sep = '\t', header = TRUE)


# -----------
# SLE data
library(stringr)
library(data.table)

sle_meta <- fread("gunzip -c ../data/SLE/cell_metadata.txt.gz")
sle_meta <- as.data.frame(sle_meta)
log2cpm <- fread("gunzip -c ../data/SLE/gene_by_cell_exp_mat.txt.gz")
log2cpm <- as.data.frame(log2cpm)
rownames(log2cpm) <- log2cpm$V1
log2cpm <- log2cpm[, -1]
dim(log2cpm)
all(sle_meta$name == colnames(log2cpm))

file_mean_sd <- "../data/sle_log2cpm_mean_sd.rds"
if (!file.exists(file_mean_sd)) {
  dat_mean_sd <- data.frame(
    mean  = Matrix::rowMeans(log2cpm),
    sd    = apply(log2cpm, 1, sd),
    count = apply(log2cpm, 1, function(x) sum(x > 0))
  )
  xs <- dat_mean_sd$mean
  xs[xs == 0] <- 1
  dat_mean_sd$cv <- dat_mean_sd$sd / xs
  rm(xs)
  dat_mean_sd$density_cv <- with(dat_mean_sd, get_density(mean, cv))
  dat_mean_sd$density_sd <- with(dat_mean_sd, get_density(mean, sd))
  saveRDS(dat_mean_sd, file_mean_sd)
} else {
  dat_mean_sd <- readRDS(file_mean_sd)
}
top_percent <- 0.8
mat_idx <- with(
  dat_mean_sd,
  mean > quantile(mean, top_percent) &
    sd > quantile(sd, top_percent)
)

sle_samples <- log2cpm[mat_idx,]
dim(sle_samples)

# PCA
scale_rows <- function(x) t(scale(t(x)))
pca <- prcomp(~ ., as.data.frame(scale_rows(sle_samples)), scale = TRUE, center = TRUE)
sle_meta <- cbind.data.frame(sle_meta, pca$rotation[, 1:100])

amp_sle_cluster_label <- sle_meta$cluster
amp_sle_cluster_label <- as.data.frame(amp_sle_cluster_label)
write.table(amp_sle_cluster_label, file = "data/amp_sle_cluster_celllabel.tsv", sep = "\t", row.names = FALSE)

amp_sle_cluster_label$amp_sle_cluster_label <- as.integer(amp_sle_cluster_label$amp_sle_cluster_label)
colnames(amp_sle_cluster_label) <- "Cluster"
write.table(amp_sle_cluster_label, file = "data/amp_sle_cluster_integerlabel.tsv", sep = "\t", row.names = FALSE)

amp_sle_pca100 <- sle_meta[, c(27:126)]
write.table(amp_sle_pca100, file = "data/amp_sle_pca100_cell.tsv", sep = "\t", row.names = FALSE)

rownames(amp_sle_pca100) <- seq(1, nrow(amp_sle_pca100))
write.table(amp_sle_pca100, file = "data/amp_sle_pca100_integer.tsv", sep = "\t", row.names = FALSE)

# -----------
# Scale RA and SLE together

# Scale RA to 0 ~ 1
ra_scale <- apply(as.matrix(ra_samples), MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
max(ra_scale)
dim(ra_scale)

# Scale SLE to 0 ~ 1
sle_scale <- apply(as.matrix(sle_samples), MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
max(sle_scale)
dim(sle_scale)

# Put the scaled datasets based on the same genes
overlap_gene <- intersect(rownames(ra_scale), rownames(sle_scale))
ra_scale_inter <- ra_scale[which(rownames(ra_scale) %in% overlap_gene),]
sle_scale_inter <- sle_scale[which(rownames(sle_scale) %in% overlap_gene),]
all(rownames(ra_scale_inter) == rownames(sle_scale_inter))
dim(ra_scale_inter)
dim(sle_scale_inter)

scale_ra_sle <- cbind.data.frame(ra_scale_inter, sle_scale_inter)
dim(scale_ra_sle)

# PCA on whole dataset
pca <- prcomp(~ ., scale_ra_sle, scale = TRUE, center = TRUE)
ra_sle_pca100 <- pca$rotation[, 1:100]
dim(ra_sle_pca100)
cell_names <- rownames(ra_sle_pca100)
rownames(ra_sle_pca100) <- seq(1, nrow(ra_sle_pca100))

colnames(sle_meta)[1] <- "cell_name"
ra_sle_cluster_label <- rbind(ra_meta[, c("cell_name", "cluster")], sle_meta[, c("cell_name", "cluster")])
dim(ra_sle_cluster_label)

all(ra_sle_cluster_label$cell_name == cell_names)

write.table(ra_sle_pca100, file = "../data/ra_sle_pca100.tsv", sep = "\t", row.names = FALSE)
write.table(ra_sle_cluster_label, file = "../data/ra_sle_cluster_label.tsv", sep = "\t", row.names = FALSE)



# -----------
# Satija data

satija_exp <- readRDS("../data/Satija_exprs_raw_sparse.rds")
satija_meta = data.frame(
  cell = as.character(colnames(satija_exp))
)
dim(satija_exp)

# Number of genes detected > 500
nGene <- Matrix::colSums(satija_exp > 0)
mito_genes <- rownames(satija_exp)[grep("^MT-", rownames(satija_exp), value = FALSE, ignore.case = TRUE)]


file_mean_sd <- "../data/satija_log2cpm_mean_sd.rds"
if (!file.exists(file_mean_sd)) {
  dat_mean_sd <- data.frame(
    mean  = Matrix::rowMeans(satija_exp),
    sd    = apply(satija_exp, 1, sd),
    count = apply(satija_exp, 1, function(x) sum(x > 0))
  )
  xs <- dat_mean_sd$mean
  xs[xs == 0] <- 1
  dat_mean_sd$cv <- dat_mean_sd$sd / xs
  rm(xs)
  dat_mean_sd$density_cv <- with(dat_mean_sd, get_density(mean, cv))
  dat_mean_sd$density_sd <- with(dat_mean_sd, get_density(mean, sd))
  saveRDS(dat_mean_sd, file_mean_sd)
} else {
  dat_mean_sd <- readRDS(file_mean_sd)
}
top_percent <- 0.8
mat_idx <- with(
  dat_mean_sd,
  mean > quantile(mean, top_percent) &
    sd > quantile(sd, top_percent)
)

satija_samples <- satija_exp[mat_idx,]
all(satija_meta$cell_name == colnames(satija_samples))
dim(satija_samples)
satija_samples <- as.matrix(satija_samples)

# # PCA
# scale_rows <- function(x) t(scale(t(x)))
# # Error: protect(): protection stack overflow
# pca <- prcomp(~ ., as.data.frame(scale_rows(satija_samples)), scale = TRUE, center = TRUE)
# satija_meta <- cbind.data.frame(satija_meta, pca$rotation[, 1:100])

satija_samples <- t(satija_samples)
rownames(satija_samples) <- seq(1, nrow(satija_samples))
write.table(satija_samples, file = "../data/satija_cells.tsv", sep = "\t", row.names = FALSE)



