suppressMessages(library('nonlinearTseries'))

raw_data <- list()
data_dir <- "/Users/ninad/Documents/_CBR/Data/Blackhole_images/Rawdata"
csv_files <- list.files(data_dir, full.names = TRUE)
for (file_path in csv_files) {
  file_name <- basename(file_path)
  if (substr(file_name, 1, 3) == "sac") {
    time_series <- read.csv(file_path, header = FALSE)
    numeric_data <- as.numeric(time_series[, 1])
    raw_data[[file_name]] <- numeric_data
  }
}

results_df <- data.frame()

for (blackhole_series in names(raw_data)) {
  time_series <- raw_data[[blackhole_series]]
  emb_dim <- 0
  tau <- NULL
  print(blackhole_series)
  tryCatch(
    {
      tau.acf <- timeLag(time_series, technique = "acf", selection.method = "first.minimum", lag.max = NULL, do.plot = FALSE)
      emb_dim <- estimateEmbeddingDim(time_series, time.lag = tau.acf, max.embedding.dim = 50, do.plot = FALSE)
      
      results_df <- rbind(results_df, c(blackhole_series,emb_dim, tau.acf))
      
      cat("Done\n")
    },
    error = function(e) {
      results_df <- rbind(results_df, c(blackhole_series, 0, NA))
      cat("Error:", conditionMessage(e), "\n")
    }
  )
}

colnames(results_df) <- c("Series", "DIM", "Tau")
write.csv(results_df, file = "/Users/ninad/Documents/_CBR/Data/blackhole_ACF_50.csv", row.names = FALSE)
