#!/usr/bin/env Rscript
# Benchmark: Sun-Abraham interaction-weighted estimator (R `fixest::sunab()`)
#
# This uses fixest::sunab() with unit+time FE and unit-level clustering,
# matching the Python SunAbraham estimator's approach.
#
# Usage:
#   Rscript benchmark_sunab.R --data path/to/data.csv --output path/to/results.json

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
    stop("Usage: Rscript benchmark_sunab.R --data <path> --output <path>")
  }

  return(result)
}

config <- parse_args(args)

# Load data
message(sprintf("Loading data from: %s", config$data))
data <- fread(config$data)

# Convert first_treat to double before assigning Inf (integer column can't hold Inf)
data[, first_treat := as.double(first_treat)]
# Convert never-treated coding: first_treat=0 -> Inf (R's convention for never-treated)
data[first_treat == 0, first_treat := Inf]

# Run benchmark
message("Running Sun-Abraham estimation with fixest::sunab()...")
start_time <- Sys.time()

# Sun-Abraham with unit+time FE, clustered at unit level
# sunab(cohort, period) creates the interaction-weighted estimator
model <- feols(
  outcome ~ sunab(first_treat, time) | unit + time,
  data = data,
  cluster = ~unit
)

estimation_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

# Extract event study effects (per-relative-period IW coefficients)
es_coefs <- coef(model)
es_ses <- se(model)

# Build event study list
event_study <- list()
coef_names <- names(es_coefs)
for (i in seq_along(es_coefs)) {
  name <- coef_names[i]
  # fixest sunab names coefficients like "time::-4" or "time::2"
  event_time <- as.numeric(gsub("^time::(-?[0-9]+)$", "\\1", name))

  event_study[[length(event_study) + 1]] <- list(
    event_time = event_time,
    att = unname(es_coefs[i]),
    se = unname(es_ses[i])
  )
}

# Aggregate to get overall ATT (weighted by observation count per cell)
# aggregate() returns a matrix with columns: Estimate, Std. Error, t value, Pr(>|t|)
agg_result <- aggregate(model, agg = "ATT")

overall_att <- agg_result[1, "Estimate"]
overall_se <- agg_result[1, "Std. Error"]
overall_pvalue <- agg_result[1, "Pr(>|t|)"]

message(sprintf("Overall ATT: %.6f (SE: %.6f)", overall_att, overall_se))

# Format output
results <- list(
  estimator = "fixest::sunab()",
  cluster = "unit",

  # Overall ATT (aggregated)
  overall_att = overall_att,
  overall_se = overall_se,
  overall_pvalue = overall_pvalue,

  # Event study effects
  event_study = event_study,

  # Timing
  timing = list(
    estimation_seconds = estimation_time,
    total_seconds = estimation_time
  ),

  # Metadata
  metadata = list(
    r_version = R.version.string,
    fixest_version = as.character(packageVersion("fixest")),
    n_units = length(unique(data$unit)),
    n_periods = length(unique(data$time)),
    n_obs = nrow(data),
    n_event_study_coefs = length(es_coefs)
  )
)

# Write output
message(sprintf("Writing results to: %s", config$output))
write_json(results, config$output, auto_unbox = TRUE, pretty = TRUE, digits = 15)

message(sprintf("Completed in %.3f seconds", estimation_time))
