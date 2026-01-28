
##################### Figure 2 PCA
# scripts/plots.R

library(tidyverse)
library(MASS)
library(ggplot2)

# prep_pca_matrix <- function(df, feature_cols, regress_age = TRUE, var_quantile = 0.2) {
#   
#   mat <- df |> dplyr::select(dplyr::all_of(feature_cols)) |> as.matrix()
#   
#   # Remove near-zero variance features
#   sds <- apply(mat, 2, sd, na.rm = TRUE)
#   keep <- sds > quantile(sds, var_quantile, na.rm = TRUE)
#   mat <- mat[, keep, drop = FALSE]
#   
#   # Regress out age if requested
#   if (regress_age) {
#     age <- df$Age
#     mat <- apply(mat, 2, function(x) {
#       resid(lm(x ~ age))
#     })
#   }
#   
#   scale(mat)
# }



# ---------------------------
# PCA runner
# ---------------------------
run_pca <- function(df, feature_cols) {
  mat <- df |>
    dplyr::select(dplyr::all_of(feature_cols)) |>
    as.matrix()

  pca <- prcomp(mat, scale. = TRUE)

  list(
    scores = as.data.frame(pca$x[, 1:2]),
    pve = summary(pca)$importance[2, 1:2] * 100
  )
}
# run_pca <- function(df, feature_cols, regress_age = TRUE) {
#   
#   mat <- prep_pca_matrix(
#     df = df,
#     feature_cols = feature_cols,
#     regress_age = regress_age
#   )
#   
#   pca <- prcomp(mat, scale. = FALSE)
#   
#   list(
#     scores = as.data.frame(pca$x[, 1:4]),   # keep more PCs
#     pve = summary(pca)$importance[2, 1:4] * 100
#   )
# }
# ---------------------------
# Decision surface (LDA)
# ---------------------------
decision_grid <- function(df, class_col, n = 200) {
  lda_fit <- lda(as.formula(paste(class_col, "~ PC1 + PC2")), data = df)
  
  grid <- expand.grid(
    PC1 = seq(min(df$PC1), max(df$PC1), length.out = n),
    PC2 = seq(min(df$PC2), max(df$PC2), length.out = n)
  )
  
  grid$class <- predict(lda_fit, grid)$class
  grid
}

# ---------------------------
# PCA plot (raw)
# ---------------------------
plot_pca_raw <- function(pca_df, pve, colors, title) {
  ggplot(pca_df, aes(PC1, PC2, color = Sex_Amyloid)) +
    geom_point(size = 3, alpha = 0.85) +
    scale_color_manual(values = colors) +
    labs(
      title = title,
      x = paste0("PC1: ", round(pve[1], 1), "%"),
      y = paste0("PC2: ", round(pve[2], 1), "%")
    ) +
    theme_classic(base_size = 13) +
    theme(plot.title = element_text(face = "bold"))
}

# ---------------------------
# PCA plot with decision regions
# ---------------------------
plot_pca_decision <- function(pca_df, grid, pve, colors, title) {
  ggplot() +
    geom_tile(
      data = grid,
      aes(PC1, PC2, fill = class),
      alpha = 0.15
    ) +
    geom_point(
      data = pca_df,
      aes(PC1, PC2, color = Sex_Amyloid),
      size = 3
    ) +
    scale_color_manual(values = colors) +
    scale_fill_manual(values = colors) +
    labs(
      title = title,
      x = paste0("PC1: ", round(pve[1], 1), "%"),
      y = paste0("PC2: ", round(pve[2], 1), "%")
    ) +
    theme_classic(base_size = 13) +
    theme(plot.title = element_text(face = "bold"), legend.position = "none")
}






################## volcano plot


# library(tidyverse)
# library(ggrepel)
# library(patchwork)
# 
# # ============================================================================
# # 1. Helper Function: Calculate Stats (Modified for log2 P and No FC Cutoff)
# # ============================================================================
# calc_de_stats <- function(df, group_col, feature_cols) {
#   results <- data.frame(
#     Feature = character(),
#     logFC = numeric(),
#     P_Value = numeric(),
#     stringsAsFactors = FALSE
#   )
#   
#   for (feat in feature_cols) {
#     if(feat %in% colnames(df)) {
#       vals_n <- df[df[[group_col]] == "N", feat]
#       vals_h <- df[df[[group_col]] == "H", feat]
#       
#       if (length(vals_n) > 2 & length(vals_h) > 2) {
#         t_res <- try(t.test(vals_h, vals_n), silent=TRUE)
#         
#         if (!inherits(t_res, "try-error")) {
#           log_fc <- mean(vals_h, na.rm=TRUE) - mean(vals_n, na.rm=TRUE)
#           
#           results <- rbind(results, data.frame(
#             Feature = feat,
#             logFC = log_fc,
#             P_Value = t_res$p.value
#           ))
#         }
#       }
#     }
#   }
#   
#   # --- MODIFICATION: Y-axis is now -log2(P) ---
#   results$logP <- -log2(results$P_Value)
#   
#   # --- MODIFICATION: Color based ONLY on P-value (No FC threshold) ---
#   results$Significance <- "NS"
#   results$Significance[results$P_Value < 0.05 & results$logFC > 0] <- "Up"
#   results$Significance[results$P_Value < 0.05 & results$logFC < 0] <- "Down"
#   
#   return(results)
# }
# 
# # ============================================================================
# # 3. Plotting Function
# # ============================================================================
# plot_volcano <- function(data, title, up_color, down_color) {
#   
#   top_features <- data %>%
#     filter(P_Value < 0.05) %>%
#     arrange(P_Value) %>%
#     head(10)
#   
#   y_cutoff <- -log2(0.05) 
#   
#   ggplot(data, aes(x = logFC, y = logP, color = Significance)) +
#     geom_point(alpha = 0.7, size = 1.8) +
#     
#     geom_hline(yintercept = y_cutoff, linetype = "dashed", color = "gray60") +
#     geom_vline(xintercept = 0, linetype = "solid", color = "gray80") +
#     
#     scale_color_manual(values = c(
#       "Up" = up_color,
#       "Down" = down_color,
#       "NS" = "gray85"
#     )) +
#     
#     geom_text_repel(
#       data = top_features,
#       aes(label = Feature),
#       size = 3,
#       max.overlaps = 15,
#       box.padding = 0.4
#     ) +
#     
#     labs(
#       title = title,
#       x = "Log2 Fold Change",
#       y = "-Log2 P-Value"
#     ) +
#     theme_classic(base_size = 12) +
#     theme(
#       legend.position = "none",
#       plot.title = element_text(face = "bold")
#     )
# }

library(tidyverse)
library(ggrepel)
library(patchwork)

# ============================================================================
# 1. Helper Function: Calculate Stats
# ============================================================================
calc_de_stats <- function(df, group_col, feature_cols) {
  results <- data.frame(
    Feature = character(),
    logFC = numeric(),
    P_Value = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (feat in feature_cols) {
    if(feat %in% colnames(df)) {
      vals_n <- df[df[[group_col]] == "N", feat]
      vals_h <- df[df[[group_col]] == "H", feat]
      
      if (length(vals_n) > 2 & length(vals_h) > 2) {
        t_res <- try(t.test(vals_h, vals_n), silent=TRUE)
        
        if (!inherits(t_res, "try-error")) {
          log_fc <- mean(vals_h, na.rm=TRUE) - mean(vals_n, na.rm=TRUE)
          
          results <- rbind(results, data.frame(
            Feature = feat,
            logFC = log_fc,
            P_Value = t_res$p.value
          ))
        }
      }
    }
  }
  
  results$logP <- -log2(results$P_Value)
  
  results$Significance <- "NS"
  results$Significance[results$P_Value < 0.05 & results$logFC > 0] <- "Up"
  results$Significance[results$P_Value < 0.05 & results$logFC < 0] <- "Down"
  
  return(results)
}

# ============================================================================
# 2. Feature Selection Function
# ============================================================================
# select_top_features <- function(data, n_up = 5, n_down = 5) {
#   # Get top upregulated features (most positive logFC with P < 0.05)
#   top_up <- data %>%
#     filter(P_Value < 0.05, logFC > 0) %>%
#     arrange(desc(logFC)) %>%
#     head(n_up)
#   
#   # Get top downregulated features (most negative logFC with P < 0.05)
#   top_down <- data %>%
#     filter(P_Value < 0.05, logFC < 0) %>%
#     arrange(logFC) %>%
#     head(n_down)
#   
#   # Combine and return
#   rbind(top_up, top_down)
# }
# 
# # ============================================================================
# # 3. Plotting Function (Improved Labeling)
# # ============================================================================
# plot_volcano <- function(data, title, up_color, down_color, 
#                          n_up = 5, n_down = 5) {
#   
#   # Select features to label based on extreme fold changes
#   top_features <- select_top_features(data, n_up, n_down)
#   
#   y_cutoff <- -log2(0.05) 
#   
#   ggplot(data, aes(x = logFC, y = logP, color = Significance)) +
#     geom_point(alpha = 0.7, size = 1.8) +
#     
#     geom_hline(yintercept = y_cutoff, linetype = "dashed", color = "gray60") +
#     geom_vline(xintercept = 0, linetype = "solid", color = "gray80") +
#     
#     scale_color_manual(values = c(
#       "Up" = up_color,
#       "Down" = down_color,
#       "NS" = "gray85"
#     )) +
#     
#     # Improved text labels with colored transparent segment lines
#     geom_text_repel(
#       data = top_features,
#       aes(label = Feature, segment.colour = Significance),
#       size = 3,
#       max.overlaps = 20,
#       box.padding = 1.0,
#       point.padding = 0.5,
#       segment.size = 0.3,
#       segment.alpha = 0.3,  # Transparency for the lines
#       min.segment.length = 0,
#       force = 3,
#       force_pull = 0.2,
#       direction = "both",
#       seed = 42
#     ) +
#     
#     # Add scale for segment colors to match points
#     scale_discrete_manual(
#       aesthetics = "segment.colour",
#       values = c(
#         "Up" = up_color,
#         "Down" = down_color,
#         "NS" = "gray85"
#       )
#     ) +
#     
#     labs(
#       title = title,
#       x = "Log2 Fold Change",
#       y = "-Log2(P-Value)"
#     ) +
#     theme_classic(base_size = 12) +
#     theme(
#       legend.position = "none",
#       plot.title = element_text(face = "bold", hjust = 0.5)
#     )
# }

plot_volcano <- function(data, title, up_color, down_color, 
                         n_up = 5, n_down = 5) {
  
  # Select features to label based on extreme fold changes
  top_features <- select_top_features(data, n_up, n_down)
  
  y_cutoff <- -log2(0.05) 
  
  ggplot(data, aes(x = logFC, y = logP, color = Significance)) +
    geom_point(alpha = 0.7, size = 1.8) +
    
    geom_hline(yintercept = y_cutoff, linetype = "dashed", color = "gray60") +
    geom_vline(xintercept = 0, linetype = "solid", color = "gray80") +
    
    scale_color_manual(values = c(
      "Up" = up_color,
      "Down" = down_color,
      "NS" = "gray85"
    )) +
    
    # Improved text labels with better visibility and placement
    geom_text_repel(
      data = top_features,
      aes(label = Feature, segment.colour = Significance),
      size = 2.2,  
      fontface = "bold",  # Bolder text for better visibility
      max.overlaps = 20,
      box.padding = 1.1,  # Increased padding around text boxes
      point.padding = 0.35,  # Increased padding around points
      segment.size = 0.3,  # Slightly thicker lines
      segment.alpha = 0.3,  # More visible lines
      min.segment.length = 0,
      force = 3,  # Increased repulsion force
      force_pull = 0.6,  # Stronger pull towards points
      direction = "both",
      seed = 42,
      # Key additions for your requirements:
      # ylim = c(y_cutoff + 0.3, NA),  # Prevent labels from going below the line
      nudge_y = 0.2  # Push labels slightly upward
    ) +
    
    # Add scale for segment colors to match points
    scale_discrete_manual(
      aesthetics = "segment.colour",
      values = c(
        "Up" = up_color,
        "Down" = down_color,
        "NS" = "gray85"
      )
    ) +
    
    labs(
      title = title,
      x = "Log2 Fold Change",
      y = "-Log2(P-Value)"
    ) +
    theme_classic(base_size = 12) +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", hjust = 0.5)
    )
}