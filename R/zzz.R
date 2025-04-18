torch <- NULL
# data table :=
.datatable.aware <- TRUE

#' @noRd
.onLoad <- function(libname, pkgname) { # nocov start
  # CRAN OMP THREAD LIMIT
  Sys.setenv("OMP_THREAD_LIMIT" = 1)
  if (is_mac()) {
    # Workaround for R's built-in OpenMP conflicts
    Sys.setenv(KMP_DUPLICATE_LIB_OK = 'TRUE')
  }
  reticulate::use_virtualenv("r-pangoling", required = FALSE)

  # use superassignment to update global reference
  torch <<- reticulate::import("torch", delay_load = TRUE, convert = FALSE)
  # TODO message or something if it's not installed
  # ask about the env
  op <- options()
  op.pangoling <- list(
    pangoling.debug = FALSE,
    pangoling.verbose = 2,
    pangoling.log.p = TRUE,
    pangoling.cache = cachem::cache_mem(max_size = 1024 * 1024^2),
    pangoling.causal.default = "gpt2",
    pangoling.masked.default = "bert-base-uncased"
  )
  toset <- !(names(op.pangoling) %in% names(op))
  if (any(toset)) options(op.pangoling[toset])

  # caching:
  tokenizer <<- memoise::memoise(tokenizer)
  lang_model <<- memoise::memoise(lang_model)
  transformer_vocab <<- memoise::memoise(transformer_vocab)
  
  # avoid notes:
  utils::globalVariables(c("mask_n","pred"))

  invisible()
} # nocov end

.onAttach <- function(libname, pkgname) {
  inst_msg <- ""
  if(installed_py_pangoling() == FALSE) {
    inst_msg <- paste0("The package needs some python dependencies, ", 
                       "to install them use `install_py_pangoling()`\n")
  }
  packageStartupMessage(pkgname,
                        " version ",
                        utils::packageVersion(pkgname),
                        inst_msg,
                        "\nAn introduction to the package can be found in ",
                        "https://docs.ropensci.org/pangoling/articles/\n",
                        "Notice that pretrained models and tokenizers are ",
                        "downloaded from https://huggingface.co/ the first ",
                        "time they are used.\n",
                        "For changing the cache folder use:\n",
                        "set_cache_folder(my_new_path)")
}

is_mac <- function(){
  grepl("darwin", R.Version()$platform)
}
