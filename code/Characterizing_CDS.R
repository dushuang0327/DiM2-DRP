data_dir <- "D:/work2/DeepCCDS-main/data"
memory.limit(size = 64000)
chr_dir  <- file.path(data_dir, "Characterizing_CDS")

gmt_kegg    <- file.path(chr_dir, "c2_kegg_symbol.gmt")
gmt_reactome<- file.path(chr_dir, "c2_reactome_symbol.gmt")

PERM_B   <- 600L     # 可先 300 调试，最终 ≥600
STAB_R   <- 40L      # 稳定性抽样轮数（80%样本/每轮）

# 双通道随机游走参数
ALPHA    <- 0.75

# 候选筛选与最终规模
FDR_CUTOFF <- 0.10
MIN_SET <- 10; MAX_SET <- 500
FINAL_K <- 38


suppressWarnings(dir.create("C:/Rtmp", showWarnings = FALSE, recursive = TRUE))
Sys.setenv(TMPDIR="C:/Rtmp", TMP="C:/Rtmp", TEMP="C:/Rtmp",
           HDF5_USE_FILE_LOCKING="FALSE", USE_GPU="FALSE")
options(HDF5Array.dump.dir="C:/Rtmp", expressions=5e5, memory.limit=56000)

pkg_ok <- function(p) suppressWarnings(requireNamespace(p, quietly=TRUE))
cran_install <- function(pkgs){ need <- pkgs[!vapply(pkgs, pkg_ok, logical(1))]
  if(length(need)) install.packages(need, repos="https://cloud.r-project.org") }
cran_install(c("data.table","Matrix","igraph","dplyr","limma","withr"))

if(!pkg_ok("BiocManager")) install.packages("BiocManager", repos="https://cloud.r-project.org")
r_major <- as.integer(R.version$major); r_minor <- as.numeric(strsplit(R.version$minor,"\\.")[[1]][1])
if(r_major==4L && r_minor==4) BiocManager::install(version="3.19", ask=FALSE, update=FALSE)
if(r_major==4L && r_minor==3) BiocManager::install(version="3.18", ask=FALSE, update=FALSE)
BiocManager::install(c("HDF5Array","DelayedArray","rhdf5","GSVA","SummarizedExperiment","BiocParallel"),
                     ask=FALSE, update=FALSE)

suppressPackageStartupMessages({
  library(HDF5Array); HDF5Array::setHDF5DumpDir("C:/Rtmp")
  library(data.table); library(Matrix); library(GSVA); library(limma); library(dplyr); library(BiocParallel)
})
register(BiocParallel::SerialParam())   # 单线程


must_files <- c(
  file.path(chr_dir, "model_list_20230923.csv"),
  file.path(chr_dir, "gene_identifiers_20191101.csv"),
  file.path(chr_dir, "rnaseq_tpm_20220624.csv"),
  file.path(data_dir, "GDSC_train_IC50_by_borh_cv00.csv"),
  gmt_kegg
)
nf <- must_files[!file.exists(must_files)]
if(length(nf)) stop("缺少文件：\n - ", paste(nf, collapse="\n - "))

GDSC_model <- read.csv(file.path(chr_dir, "model_list_20230923.csv"),
                       header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
GDSC_model <- GDSC_model[, c(1,5,7,10,13)]
GDSC_model <- GDSC_model[GDSC_model[,5]=="Cell Line", ]
GDSC_model <- GDSC_model[GDSC_model[,4] %in% c("Tumour","Metastasis"), ]
colnames(GDSC_model) <- c("model_id","project","disease_status","tissue","specimen_type")

gene_annot <- fread(file.path(chr_dir, "gene_identifiers_20191101.csv"),
                    data.table=FALSE, encoding="UTF-8")

if (!all(c("ensembl_gene_id", "hgnc_symbol") %in% colnames(gene_annot))) {
  stop(" 缺少 ensembl_gene_id 或 hgnc_symbol 列，请检查 gene_identifiers_20191101.csv")
}

ens2sym <- unique(gene_annot[, c("gene_id", "hgnc_symbol")])
colnames(ens2sym) <- c("sidg", "symbol")
ens2sym <- ens2sym[!is.na(ens2sym$symbol) & ens2sym$symbol != "", ]

# 读表达
raw_exp_all <- fread(file.path(chr_dir, "rnaseq_tpm_20220624.csv"), data.table=FALSE)
raw_exp <- raw_exp_all[-c(2,3,4,5), ]
colnames(raw_exp) <- as.character(unlist(raw_exp[1, ]))
colnames(raw_exp)[1] <- "gene_id"
raw_exp <- raw_exp[-1, ]
keep_cols <- c("gene_id", intersect(colnames(raw_exp), GDSC_model$model_id))
raw_exp <- raw_exp[, keep_cols, drop=FALSE]


GDSC_exp <- dplyr::inner_join(ens2sym, raw_exp, by = c("sidg" = "gene_id"))

tf <- tempfile(fileext = ".txt")
write.table(GDSC_exp, tf, sep = "\t", quote = FALSE, row.names = FALSE)
GDSC_exp <- read.table(tf, sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
unlink(tf)


num_cols <- setdiff(colnames(GDSC_exp), c("sidg", "symbol"))
GDSC_exp[num_cols] <- lapply(GDSC_exp[num_cols], function(x) {
  x <- trimws(x)
  x <- gsub(",", "", x)
  suppressWarnings(as.numeric(x))
})


if ("symbol" %in% colnames(GDSC_exp)) {
  rownames(GDSC_exp) <- GDSC_exp$symbol
  GDSC_exp <- GDSC_exp[, setdiff(colnames(GDSC_exp), "symbol"), drop = FALSE]
}


GDSC_exp <- as.matrix(GDSC_exp)
mode(GDSC_exp) <- "numeric"


GDSC_exp <- GDSC_exp[rowSums(GDSC_exp) > 0, , drop = FALSE]


sym_map <- rownames(GDSC_exp)
GDSC_exp <- rowsum(GDSC_exp, group = sym_map)
GDSC_exp <- sweep(GDSC_exp, 1,
                  as.numeric(table(sym_map[match(rownames(GDSC_exp),
                                                 names(table(sym_map)))])), "/")


GDSC_exp <- as.matrix(GDSC_exp)
mode(GDSC_exp) <- "numeric"



GDSC_exp <- log2(GDSC_exp + 1)


common_cells <- intersect(colnames(GDSC_exp), GDSC_model$model_id)
GDSC_exp <- GDSC_exp[, common_cells, drop = FALSE]
GDSC_exp <- t(scale(t(GDSC_exp))); GDSC_exp[is.na(GDSC_exp)] <- 0
GDSC_exp <- as.data.frame(GDSC_exp)
cat("表达矩阵（基因×样本）：", nrow(GDSC_exp), "×", ncol(GDSC_exp), "\n")


## ===================== 3) IC50 标签（四分位二分类） ===================== ##
ic50 <- fread(file.path(data_dir, "GDSC_train_IC50_by_borh_cv00.csv"))
map_df <- unique(GDSC_model[, c("model_id","project")])

fix_dash <- function(x) gsub("[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]", "-", x)
ic50[, cell_name := trimws(fix_dash(cell_name))]
map_df$project <- trimws(fix_dash(map_df$project))

ic50 <- merge(ic50, map_df, by.x="cell_name", by.y="project", all.x=TRUE)
ic50 <- ic50[!is.na(model_id)]
cell_ic50 <- ic50[, .(cell_ic50 = median(IC50, na.rm=TRUE)), by=.(model_id)]
cell_ic50 <- as.data.frame(cell_ic50)
cell_ic50 <- cell_ic50[cell_ic50$model_id %in% colnames(GDSC_exp), ]
stopifnot(nrow(cell_ic50)>0)

lo <- quantile(cell_ic50$cell_ic50, 0.25, na.rm=TRUE)
hi <- quantile(cell_ic50$cell_ic50, 0.75, na.rm=TRUE)
cell_ic50$label <- NA_integer_
cell_ic50$label[cell_ic50$cell_ic50 <= lo] <- 1
cell_ic50$label[cell_ic50$cell_ic50 >= hi] <- 0
lab_cells <- dplyr::filter(cell_ic50, !is.na(label)) %>% dplyr::pull(model_id)
cat(sprintf("标签样本：%d\n", length(lab_cells)))

## ===================== 4) limma t → z，双通道RWR ===================== ##
rwr <- function(L, s, alpha=0.75, tol=1e-10, maxit=5000){
  if(sum(abs(s))==0) return(rep(0, length(s)))
  s1 <- s / sum(abs(s)); f <- s1
  for(it in 1:maxit){
    f_new <- (1 - alpha) * s1 + alpha * as.numeric(L %*% f)
    if(sum(abs(f_new - f)) < tol) break
    f <- f_new
  }
  f
}

compute_rwr_signal <- function(exp_mat, labels_vec, adjM, alpha=ALPHA){
  expr_mat <- as.matrix(exp_mat[, names(labels_vec), drop=FALSE])
  design <- model.matrix(~ factor(labels_vec))
  fit <- lmFit(expr_mat, design); fit <- eBayes(fit)
  tstat <- fit$t[,2]; tstat[is.na(tstat)] <- 0
  t_z <- scale(tstat); t_z <- as.numeric(t_z); names(t_z) <- rownames(expr_mat)

  # 网络对齐与规范化
  adjM <- as(Matrix::Matrix(adjM, sparse=TRUE), "CsparseMatrix")
  diag(adjM) <- 0; adjM <- (adjM + t(adjM))/2
  genes_in_net <- intersect(rownames(adjM), names(t_z))
  adjM2 <- adjM[genes_in_net, genes_in_net, drop=FALSE]
  t_z2  <- t_z[genes_in_net]
  d <- Matrix::rowSums(adjM2); d[d==0] <- 1
  D.inv.sqrt <- Diagonal(x = 1/sqrt(d))
  L <- D.inv.sqrt %*% adjM2 %*% D.inv.sqrt

  s_pos <- t_z2; s_pos[s_pos<0] <- 0
  s_neg <- t_z2; s_neg[s_neg>0] <- 0; s_neg <- -s_neg
  f_pos <- rwr(L, s_pos, alpha=alpha); f_neg <- rwr(L, s_neg, alpha=alpha)
  names(f_pos) <- genes_in_net; names(f_neg) <- genes_in_net
  f_pos - f_neg
}

## 载入PPI
ppi_rdata_guess <- c(
  file.path("C:/Users/Nek/Downloads","adjM_PPI.rdata"),
  file.path(chr_dir, "adjM_PPI.rdata"),
  file.path(data_dir, "adjM_PPI.rdata")
)
ppi_rdata <- ppi_rdata_guess[file.exists(ppi_rdata_guess)][1]
if (is.na(ppi_rdata)) stop("未找到 adjM_PPI.rdata")
load(ppi_rdata); stopifnot(exists("adjM"))

# 仅用有标签的样本计算表达信号 -> RWR
lbl <- cell_ic50$label[match(lab_cells, cell_ic50$model_id)]
names(lbl) <- lab_cells
f_net <- compute_rwr_signal(GDSC_exp, lbl, adjM, alpha=ALPHA)
cat("网络传播得分基因数：", length(f_net), "\n")


read_gmt <- function(path){
  lines <- readLines(path, warn=FALSE); lines <- lines[nzchar(lines)]
  parts <- strsplit(lines, "\t")
  sets <- lapply(parts, function(x) x[-c(1,2)])
  names(sets) <- vapply(parts, `[[`, "", 1)
  sets
}
gset_list <- list(read_gmt(gmt_kegg))
if(file.exists(gmt_reactome)) gset_list <- c(gset_list, list(read_gmt(gmt_reactome)))
gset <- do.call(c, gset_list)

# 交集并按大小过滤
gset <- lapply(gset, function(v) intersect(v, names(f_net)))
sizes <- sapply(gset, length)
gset <- gset[sizes >= MIN_SET & sizes <= MAX_SET]
if(length(gset) < 100) stop("可用通路不足（<100），请检查基因名一致性。")
cat(✅ 候选集合（未富集前规模）：", length(gset), "\n")

rank_vec <- rank(f_net, ties.method="average")
rank_vec <- (rank_vec - mean(rank_vec))/sd(rank_vec)
names(rank_vec) <- names(f_net)

es_fun <- function(genes){
  m <- mean(rank_vec[genes])
  s <- sign(mean(f_net[genes]))
  s * m
}
set_names <- names(gset)
ES <- vapply(gset, es_fun, numeric(1)); names(ES) <- set_names

set.seed(1L)
ES_null <- matrix(0, nrow=length(gset), ncol=PERM_B)
rv_base <- rank_vec
for(b in 1:PERM_B){
  rv <- sample(rv_base, length(rv_base), replace=FALSE)
  ES_null[,b] <- vapply(gset, function(genes){
    s <- sign(mean(f_net[genes]))
    s * mean(rv[genes])
  }, numeric(1))
}
p_two <- vapply(seq_along(ES), function(i) mean(abs(ES_null[i,]) >= abs(ES[i])), numeric(1))
fdr <- p.adjust(p_two, method="fdr")

peanut_tbl <- data.frame(pathway=set_names, ES=ES, pval=p_two, FDR=fdr,
                         size=sapply(gset, length), row.names=NULL)
peanut_tbl <- peanut_tbl[order(peanut_tbl$FDR, -abs(peanut_tbl$ES)), ]

# 初筛：FDR≤0.1，保证100–200条；不足则按FDR排序补齐到200
cand_tbl <- subset(peanut_tbl, FDR <= FDR_CUTOFF)
if(nrow(cand_tbl) < 100){
  fill_n <- min(200, nrow(peanut_tbl))
  cand_tbl <- peanut_tbl[1:fill_n, ]
} else if(nrow(cand_tbl) > 200){
  cand_tbl <- cand_tbl[1:200, ]
}
gset_cand <- gset[cand_tbl$pathway]
cat("初筛候选通路数：", length(gset_cand), "\n")


stability_count <- setNames(numeric(length(gset_cand)), names(gset_cand))
for(r in seq_len(STAB_R)){
  set.seed(1000 + r)
  sub_cells <- sample(lab_cells, size = floor(0.8 * length(lab_cells)), replace = FALSE)
  sub_lbl   <- cell_ic50$label[match(sub_cells, cell_ic50$model_id)]
  names(sub_lbl) <- sub_cells
  f_sub <- compute_rwr_signal(GDSC_exp, sub_lbl, adjM, alpha=ALPHA)

  rv <- rank(f_sub, ties.method="average")
  rv <- (rv - mean(rv))/sd(rv); names(rv) <- names(f_sub)
  ES_sub <- vapply(gset_cand, function(genes){
    s <- sign(mean(f_sub[genes])); s * mean(rv[genes])
  }, numeric(1))
  # 近似p值（用整体空分布的分位替代，避免每轮重抽）
  p2_sub <- vapply(names(ES_sub), function(nm){
    i <- match(nm, peanut_tbl$pathway)
    null_i <- ES_null[i,]
    mean(abs(null_i) >= abs(ES_sub[nm]))
  }, numeric(1))
  fdr_sub <- p.adjust(p2_sub, method="fdr")
  sel <- names(ES_sub)[fdr_sub <= FDR_CUTOFF]
  stability_count[sel] <- stability_count[sel] + 1
}
stab_freq <- stability_count / STAB_R
cand_tbl$stab_freq <- stab_freq[match(cand_tbl$pathway, names(stab_freq))]
cand_tbl$stab_freq[is.na(cand_tbl$stab_freq)] <- 0


# 信息量：方差 × 与IC50相关性
expr_all <- as.matrix(GDSC_exp)
register(BiocParallel::SerialParam()); gc()
gsva_cand <- gsva(expr_all, gset_cand, method="ssgsea",
                  kcdf="Gaussian", abs.ranking=TRUE, ssgsea.norm=TRUE, parallel.sz=1)
gsva_cand <- as.data.frame(gsva_cand)

rho <- rep(0, nrow(gsva_cand)); names(rho) <- rownames(gsva_cand)
ic_vec <- cell_ic50$cell_ic50[match(colnames(gsva_cand), cell_ic50$model_id)]
for(i in seq_len(nrow(gsva_cand))){
  x <- as.numeric(gsva_cand[i, ])
  ok <- which(!is.na(ic_vec))
  rho[i] <- if(length(ok) >= 10) suppressWarnings(cor(x[ok], ic_vec[ok],
                             method="spearman", use="pairwise.complete.obs")) else 0
}
info_score <- apply(gsva_cand, 1, var) * abs(rho)
cand_tbl$info_score <- info_score[match(cand_tbl$pathway, rownames(gsva_cand))]

# Jaccard相似度（按基因集合重叠）
path_names <- cand_tbl$pathway
jacc <- function(a,b){
  sa <- gset_cand[[a]]; sb <- gset_cand[[b]]
  inter <- length(intersect(sa,sb)); uni <- length(union(sa,sb))
  if(uni==0) return(0) else inter/uni
}
J <- matrix(0, length(path_names), length(path_names), dimnames=list(path_names, path_names))
for(i in seq_along(path_names)) for(j in i:length(path_names)){
  v <- jacc(path_names[i], path_names[j]); J[i,j] <- v; J[j,i] <- v
}
D <- as.dist(1 - J)
hc <- hclust(D, method="average")
# 先切成 K=FINAL_K，再每簇选代表：优先 stab_freq 高、其次 info_score、再次 |ES|
cl <- cutree(hc, k=FINAL_K)
pick_rep <- function(ix){
  sub <- cand_tbl[match(path_names[ix], cand_tbl$pathway), ]
  sub[order(-sub$stab_freq, -sub$info_score, -abs(sub$ES), sub$FDR), ][1, "pathway"]
}
rep_idx <- tapply(seq_along(cl), cl, pick_rep)
sel_names <- unname(unlist(rep_idx))
if(length(sel_names) != FINAL_K){
  pool <- setdiff(path_names, sel_names)
  need <- FINAL_K - length(sel_names)
  ord_pool <- cand_tbl[match(pool, cand_tbl$pathway), ]
  ord_pool <- ord_pool[order(-ord_pool$stab_freq, -ord_pool$info_score, ord_pool$FDR), ]
  if(need > 0) sel_names <- c(sel_names, head(ord_pool$pathway, need))
  if(need < 0) sel_names <- sel_names[seq_len(FINAL_K)]
}
gset_final <- gset_cand[sel_names]
cat("最终通路数：", length(gset_final), "\n")

## ===================== 9) 计算最终38条 ssGSEA 并导出 ===================== ##
register(BiocParallel::SerialParam()); gc()
gsva_38 <- gsva(expr_all, gset_final, method="ssgsea",
                kcdf="Gaussian", abs.ranking=TRUE, ssgsea.norm=TRUE, parallel.sz=1)

GDSC_ssgsea <- as.data.frame(scale(t(gsva_38)))
GDSC_ssgsea[is.na(GDSC_ssgsea)] <- 0

GDSC_cell_line <- sort(colnames(expr_all))
GDSC_cell_line_map <- data.frame(model_id=GDSC_cell_line, stringsAsFactors=FALSE) |>
  dplyr::inner_join(GDSC_model, by="model_id") |>
  dplyr::arrange(model_id)
visible_names <- setNames(GDSC_cell_line_map$project, GDSC_cell_line_map$model_id)

idx_tbl <- data.frame(
  cell_idx = 0:(nrow(GDSC_cell_line_map)-1),
  cell_line = visible_names[GDSC_cell_line_map$model_id],
  stringsAsFactors = FALSE
)
GDSC_ssgsea <- GDSC_ssgsea[order(rownames(GDSC_ssgsea)), , drop=FALSE]


# 读取突变文件，获取有效细胞系名称
mut_path <- file.path(data_dir, "GDSC_mutation_input.csv")
mut_df <- read.csv(mut_path, stringsAsFactors = FALSE)

# 获取突变文件中的 cell_line 名称
valid_cells <- unique(trimws(mut_df$cell_line))

# 原来的映射：model_id ↔ cell_line
cell_map <- GDSC_model[, c("model_id", "project")]
colnames(cell_map) <- c("model_id", "cell_line")

# ssGSEA 的行名是 model_id
ssgsea_model_ids <- rownames(GDSC_ssgsea)

# 只保留出现在表达矩阵 AND 突变文件的细胞
cell_map <- cell_map[cell_map$model_id %in% ssgsea_model_ids, ]
cell_map <- cell_map[cell_map$cell_line %in% valid_cells, ]

# 构建 cell_idx
idx_tbl <- data.frame(
  cell_idx = seq_len(nrow(cell_map)) - 1,
  cell_line = cell_map$cell_line,
  model_id  = cell_map$model_id,
  stringsAsFactors = FALSE
)

# 匹配 ssGSEA 数值
expr_mat <- GDSC_ssgsea[cell_map$model_id, , drop = FALSE]

# 合并最终输出
out <- cbind(idx_tbl, expr_mat)

# 去掉双引号
out[] <- lapply(out, function(x){
  if(is.character(x)) gsub("\"", "", x) else x
})

# 写出
out_path <- file.path(data_dir, "GDSC_ssgsea_input.csv")
write.table(out, out_path, sep=",", row.names=FALSE, col.names=TRUE, quote=FALSE)

cat("导出完成\n")
cat("➡ 输出文件：", out_path, "\n")

