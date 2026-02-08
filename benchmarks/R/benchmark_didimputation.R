#!/usr/bin/env Rscript
# Benchmark: Imputation DiD Estimator (R `didimputation` package)
#
# Compares against diff_diff.ImputationDiD (Borusyak, Jaravel & Spiess 2024).
#
# Usage:
#   Rscript benchmark_didimputation.R --data path/to/data.csv --output path/to/results.json

library(didimputation)
library(fixest)
library(jsonlite)
library(data.table)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  result <- list(
    data = NULL,
    output = NULL
  )

  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--data") {
      result$data <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--output") {
      result$output <- args[i + 1]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }

  if (is.null(result$data) || is.null(result$output)) {
    stop("Usage: Rscript benchmark_didimputation.R --data <path> --output <path>")
  }

  return(result)
}

config <- parse_args(args)

# Load data
message(sprintf("Loading data from: %s", config$data))
data <- fread(config$data)

# Ensure proper column types
data[, unit := as.integer(unit)]
data[, time := as.integer(time)]

# R's didimputation package expects first_treat=0 or NA for never-treated units
# Our Python implementation uses first_treat=0 for never-treated, which matches
data[, first_treat := as.integer(first_treat)]
message(sprintf("Never-treated units (first_treat=0): %d", sum(data$first_treat == 0)))

# Determine event study horizons from the data
# Compute relative time for treated units
treated_data <- data[first_treat > 0]
if (nrow(treated_data) > 0) {
  treated_data[, rel_time := time - first_treat]
  min_horizon <- min(treated_data$rel_time)
  max_horizon <- max(treated_data$rel_time)
  # Post-treatment horizons only (for event study)
  post_horizons <- sort(unique(treated_data$rel_time[treated_data$rel_time >= 0]))
  all_horizons <- sort(unique(treated_data$rel_time))
  message(sprintf("Horizon range: [%d, %d]", min_horizon, max_horizon))
  message(sprintf("Post-treatment horizons: %s", paste(post_horizons, collapse = ", ")))
}

# Run benchmark - Overall ATT (static)
message("Running did_imputation (static)...")
start_time <- Sys.time()

static_result <- did_imputation(
  data = data,
  yname = "outcome",
  gname = "first_treat",
  tname = "time",
  idname = "unit",
  cluster_var = "unit"
)

static_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
message(sprintf("Static estimation completed in %.3f seconds", static_time))

# Extract overall ATT
overall_att <- static_result$estimate[1]
overall_se <- static_result$std.error[1]
message(sprintf("Overall ATT: %.6f (SE: %.6f)", overall_att, overall_se))

# Run benchmark - Event study
message("Running did_imputation (event study)...")
es_start_time <- Sys.time()

es_result <- did_imputation(
  data = data,
  yname = "outcome",
  gname = "first_treat",
  tname = "time",
  idname = "unit",
  horizon = TRUE,
  cluster_var = "unit"
)

es_time <- as.numeric(difftime(Sys.time(), es_start_time, units = "secs"))
message(sprintf("Event study estimation completed in %.3f seconds", es_time))

total_time <- static_time + es_time

# Format event study results
event_study <- data.frame(
  event_time = as.integer(gsub("tau", "", es_result$term)),
  att = es_result$estimate,
  se = es_result$std.error
)

message("Event study effects:")
for (i in seq_len(nrow(event_study))) {
  message(sprintf("  h=%d: ATT=%.4f (SE=%.4f)",
    event_study$event_time[i],
    event_study$att[i],
    event_study$se[i]))
}

# Format output
results <- list(
  estimator = "didimputation::did_imputation",

  # Overall ATT
  overall_att = overall_att,
  overall_se = overall_se,

  # Event study
  event_study = event_study,

  # Timing
  timing = list(
    static_seconds = static_time,
    event_study_seconds = es_time,
    total_seconds = total_time
  ),

  # Metadata
  metadata = list(
    r_version = R.version.string,
    didimputation_version = as.character(packageVersion("didimputation")),
    n_units = length(unique(data$unit)),
    n_periods = length(unique(data$time)),
    n_obs = nrow(data)
  )
)

# Write output
message(sprintf("Writing results to: %s", config$output))
dir.create(dirname(config$output), recursive = TRUE, showWarnings = FALSE)
write_json(results, config$output, auto_unbox = TRUE, pretty = TRUE, digits = 10)

message(sprintf("Completed in %.3f seconds", total_time))
