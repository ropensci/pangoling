
#' Set cache folder for HuggingFace transformers
#'
#' This function sets the cache directory for HuggingFace transformers. If a 
#' path is given, the function checks if the directory exists and then sets the 
#' `HF_HOME` environment variable to this path.
#' If no path is provided, the function checks for the existing cache directory 
#' in a number of environment variables.
#' If none of these environment variables are set, it provides the user with 
#' information on the default cache directory.
#'
#' @param path Character string, the path to set as the cache directory. 
#'             If NULL, the function will look for the cache directory in a 
#'             number of environment variables. Default is NULL.
#'
#' @return Nothing is returned, this function is called for its side effect of 
#'        setting the `HF_HOME` environment variable, or providing 
#'        information to the user.
#' @export
#'
#' @examplesIf installed_py_pangoling()
#' \dontrun{
#' set_cache_folder("~/new_cache_dir")
#' }
#' @seealso 
#' [Installation docs](https://huggingface.co/docs/transformers/installation?highlight=transformers_cache#cache-setup)
#' @family helper functions
set_cache_folder <- function(path = NULL) {
  if(!is.null(path)) {
    if(!dir.exists(path)) stop2("Folder '", path, "' doesn't exist.")
    reticulate::py_run_string(
                  paste0("import os\nos.environ['HF_HOME']='",
                         path,"'"))
    # reticulate::py_run_string(
    #               paste0("import os\nos.environ['HF_HOME']='",path,"'"))
  }
  path <- c(Sys.getenv("TRANSFORMERS_CACHE"),
            Sys.getenv("HUGGINGFACE_HUB_CACHE"),
            Sys.getenv("HF_HOME"),
            Sys.getenv("XDG_CACHE_HOME"))

  path <- paste0(path[path!=""],"")[1]
  if(path != ""){
    message_verbose("Pretrained models and tokenizers are downloaded ",
                    " and locally cached at '", path,"'.")
  } else {
    message_verbose(
      "By default pretrained models are downloaded and locally",
      " cached at: ~/.cache/huggingface/hub. ",
      "This is the default directory given by the shell ",
      "environment variable HF_HOME. On Windows, ",
      "the default directory is given by ",
      "C:\\Users\\username\\.cache\\huggingface\\hub.\n",
      "For changing the shell environment variables that ",
      "affect the cache folder see ",
      "https://huggingface.co/docs/transformers/installation?highlight=transformers_cache#cache-setup")
  }

}


#' List locally cached transformer models
#'
#' Returns the names of transformer models that have been downloaded and cached
#' locally by the HuggingFace `transformers` library.
#'
#' The cache directory is resolved using the following environment variables,
#' in order of precedence:
#' \enumerate{
#'   \item `HUGGINGFACE_HUB_CACHE` or `TRANSFORMERS_CACHE` — direct path to
#'         the hub cache.
#'   \item `HF_HOME` — parent directory; the hub cache is at
#'         `{HF_HOME}/hub`.
#'   \item `XDG_CACHE_HOME` — general XDG cache root; the hub cache is at
#'         `{XDG_CACHE_HOME}/huggingface/hub`.
#'   \item Default: `~/.cache/huggingface/hub`.
#' }
#'
#' @return A character vector of model names (e.g. `"gpt2"`,
#'   `"bert-base-uncased"`, `"openai-community/gpt2"`), or an empty character
#'   vector if no models are cached or the cache directory does not exist.
#'
#' @examples
#' cached_models()
#'
#' @seealso [set_cache_folder()], [remove_cached_model()],
#'   [causal_preload()], [masked_preload()]
#' @family helper functions
#' @export
cached_models <- function() {
  cache_dir <- .hf_cache_dir()

  if (!dir.exists(cache_dir)) {
    message_verbose("Cache directory '", cache_dir, "' not found.")
    return(character(0))
  }

  entries <- list.dirs(cache_dir, full.names = FALSE, recursive = FALSE)
  model_dirs <- entries[startsWith(entries, "models--")]

  if (length(model_dirs) == 0) return(character(0))

  # "models--org--name" -> "org/name", "models--name" -> "name"
  model_names <- sub("^models--", "", model_dirs)
  gsub("--", "/", model_names)
}

#' Remove a locally cached transformer model
#'
#' Deletes a model from the local HuggingFace cache. Use [cached_models()] to
#' see which models are currently cached.
#'
#' @param model Character string. The model name to remove, exactly as returned
#'   by [cached_models()] (e.g. `"gpt2"`, `"bert-base-uncased"`,
#'   `"openai-community/gpt2"`).
#'
#' @return Invisibly returns the path that was deleted, or `NULL` if the model
#'   was not found.
#'
#' @examples
#' \dontrun{
#' remove_cached_model("gpt2")
#' }
#'
#' @seealso [cached_models()], [set_cache_folder()]
#' @family helper functions
#' @export
remove_cached_model <- function(model) {
  cache_dir <- .hf_cache_dir()
  dir_name <- paste0("models--", gsub("/", "--", model))
  model_path <- file.path(cache_dir, dir_name)

  if (!dir.exists(model_path)) {
    stop2("Model '", model, "' not found in cache. ",
          "Use cached_models() to see what is available.")
  }

  unlink(model_path, recursive = TRUE)
  message_verbose("Removed cached model '", model, "' from '", model_path, "'.")
  invisible(model_path)
}

#' @noRd
.hf_cache_dir <- function() {
  cache_dir <- Sys.getenv("HUGGINGFACE_HUB_CACHE", unset = "")
  if (cache_dir == "") cache_dir <- Sys.getenv("TRANSFORMERS_CACHE", unset = "")
  if (cache_dir == "") {
    hf_home <- Sys.getenv("HF_HOME", unset = "")
    if (hf_home != "") cache_dir <- file.path(hf_home, "hub")
  }
  if (cache_dir == "") {
    xdg <- Sys.getenv("XDG_CACHE_HOME", unset = "")
    if (xdg != "") cache_dir <- file.path(xdg, "huggingface", "hub")
  }
  if (cache_dir == "") cache_dir <- path.expand("~/.cache/huggingface/hub")
  cache_dir
}
