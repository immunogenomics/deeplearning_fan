setwd("/Users/fanzhang/Documents/GitHub/deeplearning_fan/")
source("../amp_phase1_ra/R/pure_functions.R")

# -------------------------
# Load single-cell data
log2cpm <- readRDS("../amp_phase1_ra/data/celseq_synovium_log2_5265cells_paper.rds")
sc_meta <- readRDS("../amp_phase1_ra//data/celseq_synovium_meta_5265cells_paper.rds")
all(colnames(log2cpm) == sc_meta$cell_name)

# Load bulk RNA-seq data
log2tpm <- readRDS("../amp_phase1_ra/data/filtered_log2tpm_lowinput_phase_1.rds")
bulk_meta <- readRDS("../amp_phase1_ra/data/filtered_meta_lowinput_phase_1.rds")
all(colnames(log2tpm) == bulk_meta$Sample.ID)

# Take overlapped genes
file_mean_sd <- "celseq_synovium_log2cpm_mean_sd.rds"
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
top_percent <- 0.75
mat_idx <- with(
  dat_mean_sd,
  mean > quantile(mean, top_percent) &
    sd > quantile(sd, top_percent)
)
dim(log2cpm[mat_idx,])
log2cpm <- log2cpm[mat_idx,]



file_mean_sd <- "celseq_synovium_log2tpm_mean_sd.rds"
if (!file.exists(file_mean_sd)) {
  dat_mean_sd <- data.frame(
    mean  = Matrix::rowMeans(log2tpm),
    sd    = apply(log2tpm, 1, sd),
    count = apply(log2tpm, 1, function(x) sum(x > 0))
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
top_percent <- 0.82
mat_idx <- with(
  dat_mean_sd,
  mean > quantile(mean, top_percent) &
    sd > quantile(sd, top_percent)
)
dim(log2tpm[mat_idx,])
log2tpm <- log2tpm[mat_idx,]


log2cpm_new <- log2cpm[which(rownames(log2cpm) %in% intersect(rownames(log2cpm), rownames(log2tpm))),]
log2tpm_new <- log2tpm[which(rownames(log2tpm) %in% intersect(rownames(log2cpm), rownames(log2tpm))),]
log2cpm_new <- log2cpm_new[ order(match(rownames(log2cpm_new), rownames(log2tpm_new))), ]
all(rownames(log2cpm_new) == rownames(log2tpm_new))
dim(log2tpm_new)
dim(log2cpm_new)

# log2tpm_new <- t(log2tpm_new)
# log2cpm_new <- t(log2cpm_new)

# Save as inputs for deep CCA
# write.table(log2cpm_new, file = paste0("sc_", nrow(log2cpm_new), "rows_", ncol(log2cpm_new), "cols.txt", sep=""), sep = "\t", row.names = TRUE, col.names = TRUE)
# write.table(log2tpm_new, file = paste0("bulk_", nrow(log2tpm_new), "rows_", ncol(log2tpm_new), "cols.txt", sep=""), sep = "\t", row.names = TRUE, col.names = TRUE)
# saveRDS(log2cpm_new, file = paste0("sc_", nrow(log2cpm_new), "rows_", ncol(log2cpm_new), "cols.rds", sep=""))
# saveRDS(log2tpm_new, file = paste0("sc_", nrow(log2tpm_new), "rows_", ncol(log2tpm_new), "cols.rds", sep=""))
# write.table(log2cpm_new, file = paste0("sc_", nrow(log2cpm_new), "rows_", ncol(log2cpm_new), "cols.csv", sep=""), sep = "\t", row.names = FALSE, col.names = FALSE)
# write.table(log2tpm_new, file = paste0("bulk_", nrow(log2tpm_new), "rows_", ncol(log2tpm_new), "cols.csv", sep=""), sep = "\t", row.names = FALSE, col.names = FALSE)

z <- gzfile("bulk_1422rows_167cols.csv.gz")
write.csv(log2tpm_new, z)
z <- gzfile("sc_1422rows_5265cols.csv.gz")
write.csv(log2cpm_new, z)

z <- gzfile("bulk_9950rows_167cols.csv.gz")
write.csv(log2tpm_new, z)
z <- gzfile("sc_9950rows_5265cols.csv.gz")
write.csv(log2cpm_new, z)

z <- gzfile("bulk_483rows_167cols.csv.gz")
write.csv(log2tpm_new, z)
z <- gzfile("sc_483rows_5265cols.csv.gz")
write.csv(log2cpm_new, z)

z <- gzfile("bulk_4674rows_167cols.csv.gz")
write.csv(log2tpm_new, z)
z <- gzfile("sc_4675rows_5265cols.csv.gz")
write.csv(log2cpm_new, z)

z <- gzfile("bulk_724rows_167cols.csv.gz")
write.csv(log2tpm_new, z)
z <- gzfile("sc_724rows_5265cols.csv.gz")
write.csv(log2cpm_new, z)

write.table(sc_meta, file = "meta_sc_483rows_5265cols.tsv", sep = "\t", row.names = TRUE, col.names = TRUE)
write.table(bulk_meta, file = "meta_bulk_483rows_167cols.tsv", sep = "\t", row.names = TRUE, col.names = TRUE)

write.table(sc_meta, file = "meta_sc_4675rows_5265cols.tsv", sep = "\t", row.names = TRUE, col.names = TRUE)
write.table(bulk_meta, file = "meta_bulk_4675rows_167cols.tsv", sep = "\t", row.names = TRUE, col.names = TRUE)

write.table(sc_meta, file = "meta_sc_724rows_5265cols.tsv", sep = "\t", row.names = TRUE, col.names = TRUE)
write.table(bulk_meta, file = "meta_bulk_724rows_167cols.tsv", sep = "\t", row.names = TRUE, col.names = TRUE)


# --------------------
# Visualize mu1 mu2 from Z
library(R.matlab)
res <- readMat("2018_07_25_variational_deep_cca_weiran_wang/sc_bulk_128batchsize_200maxepoch_0.1dropprob_1024_1024_1024_10_0.8train/vcca_res_classify.mat")
str(res)

z_train <- as.data.frame(res$z.train)
z_test <- as.data.frame(res$z.test)
z_test_tsne <- as.data.frame(res$z.tsne)
h1_test <- as.data.frame(res$h1.test)
h2_test <- as.data.frame(res$h2.test)
meta_trainData1 <-  sapply(res$meta.trainData1, paste0, collapse="")
meta_testData1 <-  sapply(res$meta.testData1, paste0, collapse="")

dim(z_train)
dim(z_test)
dim(z_test_tsne)
dim(h1_test)
dim(h2_test)

train = data.frame(
  latent = z_train,
  cell_name = meta_trainData1
)
train <- merge(sc_meta, train, by = "cell_name")
dim(train)

test = data.frame(
  latent = z_test,
  cell_name = meta_testData1
)
test <- merge(sc_meta, test, by = "cell_name")
dim(test)


ggplot() +
  geom_point(
    data = train,
    mapping = aes_string(x = "latent.V1", y = "latent.V2"),
    size = 2, stroke = 0.2, shape = 20
  ) +
  labs(
    x = "Latent 1",
    y = "Latent 2",
    title = ""
  ) +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "none"
    # axis.text = element_blank()
    # axis.ticks = element_blank()
    # panel.grid = element_blank()
  ) 


# tSNE
library(Rtsne)
tsne1 <- Rtsne(
  X = train[, c(42:51)],
  is_distance = FALSE,
  dims = 2,
  perplexity = 50,
  max_iter = 5000
  # theta = 0.5,
  # pca = FALSE
)
train$T1 <- tsne1$Y[,1]
train$T2 <- tsne1$Y[,2]

ggplot() +
  geom_point(
    data = train,
    mapping = aes_string(x = "T1", y = "T2", color = "cell_type"),
    size = 2, stroke = 0.2, shape = 20
  ) +
  labs(
    x = "tSNE1",
    y = "tSNE2",
    title = ""
  ) +
  theme_classic(base_size = 14) +
  theme(
    # legend.position = "none"
    # axis.text = element_blank()
    # axis.ticks = element_blank()
    # panel.grid = element_blank()
  ) 


# UMAP
library(umap)
umap()



