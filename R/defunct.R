#' @title Defunct functions in package \pkg{pangoling}.
#' @description The functions listed below are now defunct. 
#'  When possible, alternative functions with similar functionality are also mentioned.
#' @name pangoling-defunct
#' @keywords internal
NULL

#' @title Get the possible tokens and their log probabilities for each mask in a sentence
#' @description This function is defunct. Use `masked_tokens_pred_tbl()` instead.
#' @name masked_tokens_tbl-defunct
#' @seealso \code{\link{pangoling-defunct}}
#' @keywords internal
#' @return This function is defunct and returns an error.
NULL

#' @rdname pangoling-defunct
#' @section \code{masked_tokens_tbl}:
#' For \code{masked_tokens_tbl}, use \code{\link{masked_tokens_pred_tbl}}.
#' @export
masked_tokens_tbl <- function(...) {
  .Defunct(new = "masked_tokens_pred_tbl()")
}

#' @title Get the log probability of a target word (or phrase) given a left and right context
#' @description This function is defunct. Use `masked_targets_pred()` instead.
#' @name masked_lp-defunct
#' @seealso \code{\link{pangoling-defunct}}
#' @keywords internal
#' @return This function is defunct and returns an error.
NULL

#' @rdname pangoling-defunct
#' @section \code{masked_lp}:
#' For \code{masked_lp}, use \code{\link{masked_targets_pred}}.
#' @export
masked_lp <- function(...) {
  .Defunct(new = "masked_targets_pred()")
}

#' @title Get the possible next tokens and their log probabilities for its previous context
#' @description This function is defunct. Use `causal_next_tokens_pred_tbl()` instead.
#' @name causal_next_tokens_tbl-defunct
#' @seealso \code{\link{pangoling-defunct}}
#' @keywords internal
#' @return This function is defunct and returns an error.
NULL

#' @rdname pangoling-defunct
#' @section \code{causal_next_tokens_tbl}:
#' For \code{causal_next_tokens_tbl}, use \code{\link{causal_next_tokens_pred_tbl}}.
#' @return This function is defunct and returns an error.
#' @export
causal_next_tokens_tbl <- function(...) {
  .Defunct(new = "causal_next_tokens_pred_tbl()")
}

#' @title Get the log probability of each element of a vector of words (or phrases) using a causal transformer
#' @description This function is defunct. Use `causal_targets_pred()` or `causal_words_pred()` instead.
#' @name causal_lp-defunct
#' @seealso \code{\link{pangoling-defunct}}
#' @keywords internal
#' @return This function is defunct and returns an error.
NULL

#' @rdname pangoling-defunct
#' @section \code{causal_lp}:
#' For \code{causal_lp}, use \code{\link{causal_targets_pred}} or \code{\link{causal_words_pred}}.
#' @export
causal_lp <- function(...) {
  .Defunct(new = "causal_targets_pred() or causal_words_pred()")
}

#' @title Get the log probability of each token in a sentence (or group of sentences) using a causal transformer
#' @description This function is defunct. Use `causal_tokens_pred_lst()` instead.
#' @name causal_tokens_lp_tbl-defunct
#' @seealso \code{\link{pangoling-defunct}}
#' @keywords internal
#' @return This function is defunct and returns an error.
NULL

#' @rdname pangoling-defunct
#' @section \code{causal_tokens_lp_tbl}:
#' For \code{causal_tokens_lp_tbl}, use \code{\link{causal_tokens_pred_lst}}.
#' @export
causal_tokens_lp_tbl <- function(...) {
  .Defunct(new = "causal_tokens_pred_lst()")
}

#' @title Get a list of matrices with the log probabilities of possible words given their previous context using a causal transformer
#' @description This function is defunct. Use `causal_pred_mats()` instead.
#' @name causal_lp_mats-defunct
#' @seealso \code{\link{pangoling-defunct}}
#' @keywords internal
#' @return This function is defunct and returns an error.
NULL

#' @rdname pangoling-defunct
#' @section \code{causal_lp_mats}:
#' For \code{causal_lp_mats}, use \code{\link{causal_pred_mats}}.
#' @export
causal_lp_mats <- function(...) {
  .Defunct(new = "causal_pred_mats()")
}
