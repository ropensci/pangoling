#' Preloads a causal language model
#'
#' Preloads a causal language model to speed up next runs.
#' 
#' @section More details about causal models:
#' A causal language model (also called GPT-like, auto-regressive, or decoder
#' model) is a type of large language model usually used for text-generation
#' that can predict the next word (or more accurately in fact token) based
#' on a preceding context.
#'
#' If not specified, the causal model used will be the one set in the global
#' option `pangoling.causal.default`, this can be
#' accessed via `getOption("pangoling.causal.default")` (by default
#' "`r getOption("pangoling.causal.default")`"). To change the default option
#' use `options(pangoling.causal.default = "newcausalmodel")`.
#'
#' A list of possible causal models can be found in
#' [Hugging Face website](https://huggingface.co/models?pipeline_tag=text-generation).
#'
#' Using the  `config_model` and `config_tokenizer` arguments, it's possible to
#'  control how the model and tokenizer from Hugging Face is accessed, see the
#'  Python method
#'  [`from_pretrained`](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained)
#'  for details.
#'
#'  In case of errors when a new model is run, check the status of
#'  [https://status.huggingface.co/](https://status.huggingface.co/)
#'
#' @param model Name of a pre-trained model or folder. One should be able to use
#' models based on "gpt2". See 
#' [hugging face website](https://huggingface.co/models?other=gpt2).
#' @param checkpoint Folder of a checkpoint.
#' @param output_hidden_states Logical. If TRUE, the model will return hidden 
#'   states from all layers, which can be extracted using the *_layers() functions.
#'   Default is FALSE. Note: Setting this to TRUE increases memory usage.
#' @param add_special_tokens Whether to include special tokens. It has the
#'                           same default as the
#' [AutoTokenizer](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoTokenizer)
#'                            method in Python.
#' @param config_model List with other arguments that control how the
#'                      model from Hugging Face is accessed.
#' @param config_tokenizer List with other arguments that control how the 
#'                         tokenizer from Hugging Face is accessed.
#'
#' @return Nothing.
#'
#' @examplesIf installed_py_pangoling()
#' causal_preload(model = "gpt2")
#'
#' @family causal model helper functions
#' @export
#'
causal_preload <- function(model = getOption("pangoling.causal.default"),
                           checkpoint = NULL,
                           output_hidden_states = FALSE,
                           add_special_tokens = NULL,
                           config_model = NULL, config_tokenizer = NULL) {
  message_verbose("Preloading causal model ", model, "...")
  lang_model(model, 
             checkpoint = checkpoint, 
             task = "causal", 
             output_hidden_states = output_hidden_states,
             config_model = config_model)
  tokenizer(model, 
            add_special_tokens = add_special_tokens, 
            config_tokenizer = config_tokenizer)
  invisible()
}

#' Unload a causal language model and free memory
#'
#' Unloads a causal language model and its tokenizer from memory, and triggers
#' garbage collection to free up RAM and GPU memory.
#'
#' @details
#' This function helps manage memory when working with large models. It:
#' - Deletes the model and tokenizer objects from Python
#' - Empties the CUDA cache (if using GPU)
#' - Triggers garbage collection in both Python and R
#'
#' It's particularly useful when:
#' - Switching between different models
#' - Working with limited memory
#' - Running multiple models sequentially
#'
#' @return Nothing (called for side effects).
#'
#' @examplesIf installed_py_pangoling()
#' # Load a model
#' causal_preload(model = "gpt2")
#' 
#' # Do some work...
#' pred <- causal_targets_pred(contexts = "The cat", targets = "sat", model = "gpt2")
#' 
#' # Unload when done
#' causal_unload()
#'
#' @family causal model helper functions
#' @export
causal_unload <- function() {
  transformer_unload()
}

#' Returns the configuration of a causal model
#'
#' @inheritParams causal_preload
#' @inherit  causal_preload details
#' @inheritSection causal_preload More details about causal models
#' @return A list with the configuration of the model.
#' @examplesIf installed_py_pangoling()
#' causal_config(model = "gpt2")
#'
#' @family causal model helper functions
#' @export
causal_config <- function(model = getOption("pangoling.causal.default"),
                          checkpoint = NULL, config_model = NULL) {
  lang_model(
    model = model,
    checkpoint = checkpoint,
    task = "causal",
    config_model = config_model
  )$config$to_dict()
}


#' Generate next tokens after a context and their predictability using a causal 
#' transformer model
#'
#' This function predicts the possible next tokens and their predictability
#' (log-probabilities by default). The function sorts tokens in descending order
#'  of their predictability.
#'
#'
#' @details
#' The function uses a causal transformer model to compute the predictability
#' of all tokens in the model's vocabulary, given a single input context. It
#' returns a table where each row represents a token, along with its
#' predictability score. By default, the function returns log-probabilities in
#' natural logarithm (base *e*), but you can specify a different logarithm base
#' (e.g., `log.p = 1/2` for surprisal in bits).
#'
#' If `decode = TRUE`, the tokens are converted into human-readable strings,
#' handling special characters like accents and diacritics. This ensures that
#' tokens are more interpretable, especially for languages with complex 
#' tokenization.
#'
#' @param context A single string representing the context for which the next
#'                tokens and their predictabilities are predicted.
#' @param decode Logical. If `TRUE`, decodes the tokens into human-readable
#'               strings, handling special characters and diacritics. Default is
#'                `FALSE`.
#' @inheritParams causal_preload
#' @inheritParams causal_tokens_pred_lst
#' @inherit causal_preload details
#' @inheritSection causal_preload More details about causal models
#' @return A table with possible next tokens and their log-probabilities.
#' @examplesIf installed_py_pangoling()
#' causal_next_tokens_pred_tbl(
#'   context = "The apple doesn't fall far from the",
#'   model = "gpt2"
#' )
#'
#' @family causal model functions
#' @export
causal_next_tokens_pred_tbl <- 
  function(context,
           log.p = getOption("pangoling.log.p"),
           decode = FALSE,
           model = getOption("pangoling.causal.default"),
           checkpoint = NULL,
           add_special_tokens = NULL,
           config_model = NULL,
           config_tokenizer = NULL) {
    if (length(unlist(context)) > 1) {
      stop2("Only one context is allowed in this function.")
    }
    if(any(!is_really_string(context))) {
      stop2("`context` needs to contain a string.")
    }
    message_verbose_model(model, checkpoint)
    trf <- lang_model(model,
                      checkpoint = checkpoint,
                      task = "causal",
                      config_model = config_model
                      )
    tkzr <- tokenizer(model,
                      add_special_tokens = add_special_tokens,
                      config_tokenizer = config_tokenizer
                      )

    # no batches allowed
    context_tensor <- encode(list(unlist(context)),
                             tkzr,
                             add_special_tokens = add_special_tokens
                             )$input_ids
    generated_outputs <- trf(context_tensor)
    n_tokens <- length(context_tensor$tolist()[[1]]) # was 0
    logits_next_word <- generated_outputs$logits[0][n_tokens - 1]
    l_softmax <- torch$log_softmax(logits_next_word, dim = -1L)$tolist()
    lp <- reticulate::py_to_r(l_softmax) |>
      unlist()
    vocab <- get_vocab(tkzr, decode = decode)
    diff_words <- length(vocab) - length(lp)
    if(diff_words > 0) {
      warning(paste0("Tokenizer's vocabulary is longer than the model's.",
                     " Some words will have NA predictability."))
      lp <- c(lp, rep(NA, diff_words))
    } else if(diff_words < 0) {
      message_verbose("Tokenizer's vocabulary is smaller than the model's.", 
                      " The model might reserve extra embeddings for padding, dynamic additions,",
                      " or GPU efficiency.")
      vocab <- c(vocab, rep("", -diff_words))
      }
    tidytable::tidytable(token = vocab,
                         pred = lp |> ln_p_change(log.p = log.p)) |>
      tidytable::arrange(-pred)
  }



#' Compute predictability using a causal transformer model
#'
#' These functions calculate the predictability of words, phrases, or tokens 
#' using a causal transformer model. 
#'
#' @details
#' These functions calculate the predictability (by default the natural 
#' logarithm of the word probability) of words, phrases or tokens using a 
#' causal transformer model:
#' 
#' - **`causal_targets_pred()`**: Evaluates specific target words or phrases 
#'   based on their given contexts. Use when you have explicit
#'   context-target pairs to evaluate, with each target word or phrase paired 
#'   with a single preceding context. Notice that if the target is a "token"
#'   rather than a word, one needs to set `sep = ""`.
#' - **`causal_words_pred()`**: Computes predictability for all elements of a 
#'   vector grouped by a specified variable. Use when working with words or 
#'   phrases split into groups, such as sentences or paragraphs, where 
#'   predictability is computed for every word or phrase in each group. Notice 
#'   that if the elements of the vector `x` are  "tokens" rather than a words 
#'   (or phrases), one needs to set `sep = ""`.
#' - **`causal_tokens_pred_lst()`**: Computes the predictability of each token 
#'   in a sentence (or group of sentences) and returns a list of results for 
#'   each sentence. Use when you want to calculate the predictability of 
#'   every token in one or more sentences.
#'
#' See the
#' [online article](https://docs.ropensci.org/pangoling/articles/intro-gpt2.html)
#' in pangoling website for more examples.
#' 
#' @param x A character vector of words, phrases, or texts to evaluate.
#' @param word_n Word order, by default this is the word order of the vector x.
#' @param texts A vector or list of sentences or paragraphs.
#' @param targets A character vector of target words or phrases.
#' @param contexts A character vector of contexts corresponding to each target.
#' @param by A grouping variable indicating how texts are split into groups.
#' @param sep A string specifying how words are separated within contexts or 
#'            groups. Default is `" "`. For languages that don't have spaces 
#'            between words (e.g., Chinese), set `sep = ""`.
#' @param log.p Base of the logarithm used for the output predictability values.
#'              If `TRUE` (default), the natural logarithm (base *e*) is used.
#'              If `FALSE`, the raw probabilities are returned.
#'              Alternatively, `log.p` can be set to a numeric value specifying
#'              the base of the logarithm (e.g., `2` for base-2 logarithms).
#'              To get surprisal in bits (rather than predictability), set
#'              `log.p = 1/2`.
#' @param ignore_regex Can ignore certain characters when calculating the log
#'                      probabilities. For example `^[[:punct:]]$` will ignore
#'                      all punctuation  that stands alone in a token.
#' @param batch_size Maximum number of sentences/texts processed in parallel. 
#'                   Larger batches increase speed but use more memory. Since 
#'                   all texts in a batch must have the same length, shorter 
#'                   ones are padded with placeholder tokens.
#' @inheritParams causal_preload
#' @inheritSection causal_preload More details about causal models
#' @param ... Currently not in use.
#' @return For `causal_targets_pred()` and `causal_words_pred()`, 
#'   a named numeric vector of predictability scores. For 
#'   `causal_tokens_pred_lst()`, a list of named numeric vectors, one for 
#'   each sentence or group.
#'
#' @examplesIf installed_py_pangoling()
#' # Using causal_targets_pred
#' causal_targets_pred(
#'   contexts = c("The apple doesn't fall far from the",
#'                "Don't judge a book by its"),
#'   targets = c("tree.", "cover."),
#'   model = "gpt2"
#' )
#'
#' # Using causal_words_pred
#' causal_words_pred(
#'   x = df_sent$word,
#'   by = df_sent$sent_n,
#'   model = "gpt2"
#' )
#' 
#' # Using causal_tokens_pred_lst
#' preds <- causal_tokens_pred_lst(
#'   texts = c("The apple doesn't fall far from the tree.",
#'             "Don't judge a book by its cover."),
#'   model = "gpt2"
#' )
#' preds
#'
#' # Convert the output to a tidy table
#' suppressPackageStartupMessages(library(tidytable))
#' map2_dfr(preds, seq_along(preds), 
#' ~ data.frame(tokens = names(.x), pred = .x, id = .y))
#'
#' @family causal model functions
#' @export
#' @rdname causal_predictability
causal_words_pred <- function(x,
                              by = rep(1, length(x)),
                              word_n = NULL,
                              sep = " ",
                              log.p = getOption("pangoling.log.p"),
                              ignore_regex = "",
                              model = getOption("pangoling.causal.default"),
                              checkpoint = NULL,
                              add_special_tokens = NULL,
                              config_model = NULL,
                              config_tokenizer = NULL,
                              batch_size = 1,
                              ...) {
  if(any(!is_really_string(x))) {
    stop2("`x` needs to be a vector of non-empty strings.")
  }
  dots <- list(...)
  # Check for unknown arguments
  if (length(dots) > 0) {
    unknown_args <- setdiff(names(dots), ".by")
    if (length(unknown_args) > 0) {
      stop("Unknown arguments: ", paste(unknown_args, collapse = ", "), ".")
    }
  }
  if (length(x) != length(by)) {
    stop2("The argument `by` has an incorrect length.")
  }
  if(is.null(word_n)){
    word_n <- stats::ave(seq_along(by), by, FUN = seq_along)
  }
  if (length(word_n) != length(by)) {
    stop2("The argument `word_n` has an incorrect length.")
  }
  if(any(x != trimws(x)) & sep == " ") {
    message_verbose(paste0("Notice that some words have white spaces,",
                           ' argument `sep` should probably set to "".'))
  }
  
  stride <- 1 # fixed for now
  message_verbose_model(model, checkpoint = checkpoint)
  
  word_by_word_texts <- split(x, by, drop = TRUE)
  
  word_n_by <- split(word_n, by, drop = TRUE)
  word_by_word_texts <- tidytable::map2(word_by_word_texts, word_n_by, ~ .x[order(.y)])
  
  pasted_texts <- conc_words(word_by_word_texts, sep = sep)
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "causal",
                    config_model = config_model)
  tensors <- create_tensor_lst(
    texts = unname(pasted_texts),
    tkzr = tkzr,
    add_special_tokens = add_special_tokens,
    stride = stride,
    batch_size = batch_size
  )

  lmats <- lapply(tensors, function(tensor) {
    causal_mat(tensor,
               trf = trf,
               tkzr = tkzr,
               add_special_tokens = add_special_tokens,
               decode = FALSE,
               stride = stride)
  }) |>
    unlist(recursive = FALSE)

  out <- tidytable::pmap(
                      list(
                        word_by_word_texts,
                        names(word_by_word_texts),
                        lmats
                      ),
                      function(words, item, mat) {
                        # words <- word_by_word_texts[[1]]
                        # item <- names(word_by_word_texts)[[1]]
                        # mat <- lmats[[1]]

                        message_verbose(
                          "Text id: ", item, "\n`",
                          paste(words, collapse = sep),
                          "`"
                        )
                        word_lp(words,
                                sep = sep,
                                mat = mat,
                                ignore_regex = ignore_regex,
                                model = model,
                                add_special_tokens = add_special_tokens,
                                config_tokenizer = config_tokenizer)
                      }
                    )

  message_verbose("***\n")
  
  out_reordered <- tidytable::map2(out, word_n_by, 
                                   ~ .x[as.integer(as.factor(.y))])
  lps <- out_reordered |> unsplit(by, drop = TRUE)
    names(lps) <- x 
  # out |> lapply(function(x) paste0(names(x),"")) |>
  #   unsplit(by, drop = TRUE)
  lps |> ln_p_change(log.p = log.p)
}




#' @rdname causal_predictability
#' @export
causal_tokens_pred_lst <- 
  function(texts,
           log.p = getOption("pangoling.log.p"),
           model = getOption("pangoling.causal.default"),
           checkpoint = NULL,
           add_special_tokens = NULL,
           config_model = NULL,
           config_tokenizer = NULL,
           batch_size = 1) {
    if(any(!is_really_string(texts))){
      stop2("`texts` needs to be a vector of non-empty strings.")
    }
    stride <- 1
    message_verbose_model(model, checkpoint)
    ltexts <- as.list(unlist(texts, recursive = TRUE))
    tkzr <- tokenizer(model,
                      add_special_tokens = add_special_tokens,
                      config_tokenizer = config_tokenizer)
    trf <- lang_model(model,
                      checkpoint = checkpoint,
                      task = "causal",
                      config_model = config_model)
    tensors <- create_tensor_lst(ltexts,
                                 tkzr,
                                 add_special_tokens = add_special_tokens,
                                 stride = stride,
                                 batch_size = batch_size)

    ls_mat <- tidytable::map(tensors, function(tensor) {
      causal_mat(tensor,
                 trf,
                 tkzr,
                 add_special_tokens = add_special_tokens,
                 decode = FALSE,
                 stride = stride)
    }) |>
      unlist(recursive = FALSE)

    lapply(ls_mat, function(mat) {
      if (ncol(mat) == 1 && colnames(mat) == "") {
        pred <-  NA_real_
        names(pred) <- ""
      } else {
        pred <- tidytable::map2_dbl(colnames(mat),
                                   seq_len(ncol(mat)),
                                   ~ mat[.x, .y]) |>
          ln_p_change(log.p = log.p)
        names(pred) <- colnames(mat)

      }
      pred
    })
  }


#' @noRd
causal_mat <- function(tensor,
                       trf,
                       tkzr,
                       add_special_tokens = NULL,
                       decode,
                       stride = 1) {
  message_verbose(
    "Processing a batch of size ",
    tensor$input_ids$shape[0],
    " with ",
    tensor$input_ids$shape[1], " tokens."
  )

  if (tensor$input_ids$shape[1] == 0) {
    warning("No tokens found.", call. = FALSE)
    vocab <- get_vocab(tkzr, decode = decode)
    mat <- matrix(rep(NA, length(vocab)), ncol = 1)
    rownames(mat) <- vocab
    colnames(mat) <- ""
    return(list(mat))
  }

  logits_b <- trf$forward(
                    input_ids = tensor$input_ids,
                    attention_mask = tensor$attention_mask
                  )$logits

  lmat <- lapply(seq_len(logits_b$shape[0]) - 1, function(i) {
    real_token_pos <- seq_len(sum(tensor$attention_mask[i]$tolist())) - 1
    logits <- logits_b[i][real_token_pos]
    # in case it's only one token, it needs to be unsqueezed
    ids <- tensor$input_ids[i]$unsqueeze(1L)
    tokens <- tkzr$convert_ids_to_tokens(ids[real_token_pos])
    lp <- reticulate::py_to_r(torch$log_softmax(logits, dim = -1L))$tolist()
    rm(logits)
    gc(full = TRUE)
    if (is.list(lp)) {
      mat <- do.call("cbind", lp)
    } else {
      # In case it's only one token, lp won't be a list
      mat <- matrix(lp, ncol = 1)
    }
    # remove the last prediction, and the first is NA
    mat <- cbind(rep(NA, nrow(mat)), mat[, -ncol(mat)])
    # in case the last words in the vocab were not used to train the model
    vocab <- get_vocab(tkzr, decode = decode)
    diff_words <- length(vocab) - nrow(mat)
    if(diff_words > 0) {
      warning("Tokenizer's vocabulary is larger than the model's.")
    } else if(diff_words < 0) {
      message_verbose("Tokenizer's vocabulary is smaller than the model's.", 
      " The model might reserve extra embeddings for padding, dynamic additions,",
      " or GPU efficiency.")
    }
    rownames(mat) <- vocab[seq_len(nrow(mat))]
    colnames(mat) <- unlist(tokens)
    mat
  })
  rm(logits_b)
  lmat
}



#' Generate a list of predictability matrices using a causal transformer model
#'
#' This function computes a list of matrices, where each matrix corresponds to a
#' unique group specified by the `by` argument. Each matrix represents the
#' predictability of every token in the input text (`x`) based on preceding 
#' context, as evaluated by a causal transformer model.
#'
#'
#' @details
#' The function splits the input `x` into groups specified by the `by` argument 
#' and processes each group independently. For each group, the model computes
#' the predictability of each token in its vocabulary based on preceding 
#' context.
#'
#' Each matrix contains:
#' - Rows representing the model's vocabulary.
#' - Columns corresponding to tokens in the group (e.g., a sentence or
#' paragraph).
#' - By default, values in the matrices are the natural logarithm of word 
#' probabilities.
#'
#' @inheritParams causal_words_pred
#' @inheritParams causal_preload
#' @inheritParams causal_next_tokens_pred_tbl
#' @param sorted When default FALSE it will retain the order of groups we are 
#'               splitting by. When TRUE then sorted (according to `by`) list(s)
#'               are returned. 
#' @inherit  causal_preload details
#' @inheritSection causal_preload More details about causal models
#' @return A list of matrices with tokens in their columns and the vocabulary of
#'         the model in their rows
#'
#' @examplesIf installed_py_pangoling()
#' data("df_sent")
#' df_sent
#' list_of_mats <- causal_pred_mats(
#'                        x = df_sent$word,
#'                        by = df_sent$sent_n,  
#'                        model = "gpt2"
#'                 )
#'
#' # View the structure of the resulting list
#' list_of_mats |> str()
#'
#' # Inspect the last rows of the first matrix
#' list_of_mats[[1]] |> tail()
#'
#' # Inspect the last rows of the second matrix
#' list_of_mats[[2]] |> tail()
#' @family causal model functions
#' @export
#'
causal_pred_mats <- function(x,
                             by = rep(1, length(x)),
                             sep = " ",
                             log.p = getOption("pangoling.log.p"),
                             sorted = FALSE,
                             model = getOption("pangoling.causal.default"),
                             checkpoint = NULL,
                             add_special_tokens = NULL,
                             decode = FALSE,
                             config_model = NULL,
                             config_tokenizer = NULL,
                             batch_size = 1,
                             ...) {
  if(any(!is_really_string(x))) {
    stop2("`x` needs to be a vector of non-empty strings.")
  }
  dots <- list(...)
  if(any(x != trimws(x)) & sep == " ") {
    message_verbose(paste0('Notice that some words have white spaces,',
                           ' argument `sep` should probably set to "".'))
  }
  # Check for unknown arguments
  if (length(dots) > 0) {
    unknown_args <- setdiff(names(dots), ".by")
    if (length(unknown_args) > 0) {
      stop("Unknown arguments: ", paste(unknown_args, collapse = ", "), ".")
    }
  }
  stride <- 1
  message_verbose_model(model, checkpoint)
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer
                    )
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "causal",
                    config_model = config_model
                    )
  word_by_word_texts <- split(x, by)
  pasted_texts <- conc_words(word_by_word_texts, sep = sep)
  tensors <- create_tensor_lst(unname(pasted_texts),
                               tkzr,
                               add_special_tokens = add_special_tokens,
                               stride = stride,
                               batch_size = batch_size
                               )
  lmat <- tidytable::map(
                       tensors,
                       function(tensor) {
                         causal_mat(tensor,
                                    trf = trf,
                                    tkzr = tkzr,
                                    add_special_tokens = add_special_tokens,

                                    decode = decode,
                                    stride = stride
                                    )
                       }
                     )
  names(lmat) <- levels(as.factor(by))
  if(!sorted) lmat <- lmat[unique(as.factor(by))]
  lmat |>
    unlist(recursive = FALSE) |>
    ln_p_change(log.p = log.p)
}


#' @export
#' @rdname causal_predictability
causal_targets_pred <- function(contexts,
                                targets,
                                sep = " ",
                                log.p = getOption("pangoling.log.p"),
                                ignore_regex = "",
                                model = getOption("pangoling.causal.default"),
                                checkpoint = NULL,
                                add_special_tokens = NULL,
                                config_model = NULL,
                                config_tokenizer = NULL,
                                batch_size = 1,
                                ...) {
  if(any(!is_really_string(targets))) { 
    stop2("`targets` needs to be a vector of non-empty strings.")
  }
  if(any(!is_really_string(contexts))) {
    stop2("`contexts` needs to be a vector of non-empty strings.")
  }
  dots <- list(...)
  # Check for unknown arguments
  if (length(dots) > 0) {
    unknown_args <- setdiff(names(dots), ".by")
    if (length(unknown_args) > 0) {
      stop("Unknown arguments: ", paste(unknown_args, collapse = ", "), ".")
    }
  }
  if(any(targets != trimws(targets)) | 
     any(contexts != trimws(contexts)) & sep == " ") {
    message_verbose(
      paste0('Notice that some words have white spaces,',
             ' if this is intended, argument `sep` should probably set to "".'))
  }
  stride <- 1 # fixed for now
  message_verbose_model(model, checkpoint)
  x <- c(rbind(contexts, targets))
  by <- rep(seq_len(length(x)/2), each = 2)
  word_by_word_texts <- split(x, by, drop = TRUE)
  
  pasted_texts <- conc_words(word_by_word_texts, sep = sep)
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer
                    )
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "causal",
                    config_model = config_model
                    )
  tensors <- create_tensor_lst(
    texts = unname(pasted_texts),
    tkzr = tkzr,
    add_special_tokens = add_special_tokens,
    stride = stride,
    batch_size = batch_size
  )

  lmats <- lapply(tensors, function(tensor) {
    causal_mat(tensor,
               trf = trf,
               tkzr = tkzr,
               add_special_tokens = add_special_tokens,
               decode = FALSE,
               stride = stride
               )
  }) |>
    unlist(recursive = FALSE)
  out <- tidytable::pmap(
                      list(
                        word_by_word_texts,
                        names(word_by_word_texts),
                        lmats
                      ),
                      function(words, item, mat) {
                        message_verbose(
                          "Text id: ", item, "\n`",
                          paste(words, collapse = sep),
                          "`"
                        )
                        word_lp(words,
                                sep = sep,
                                mat = mat,
                                ignore_regex = ignore_regex,
                                model = model,
                                add_special_tokens = add_special_tokens,
                                config_tokenizer = config_tokenizer
                                )
                      }
                    )
  message_verbose("***\n")
  keep <- c(FALSE, TRUE)

  out <- out |> lapply(function(x) x[keep])
  lps <- out |> unsplit(by[keep], drop = TRUE)

  names(lps) <- out |> lapply(function(x) paste0(names(x),"")) |>
    unsplit(by[keep], drop = TRUE)
  lps |>
    ln_p_change(log.p = log.p)
}



#' Extract hidden layer representations using a causal transformer model
#'
#' These functions extract hidden layer representations (embeddings) from 
#' transformer models for words, phrases, or tokens.
#'
#' @details
#' These functions extract hidden states from all layers of a transformer model,
#' including layer 0 (non-contextualized token embeddings) and layers 1-N 
#' (contextualized representations from each transformer block).
#' 
#' **Layer numbering:**
#' - Layer 0: Non-contextualized token embeddings (input to the transformer)
#' - Layers 1-N: Output from each transformer block (contextualized)
#' 
#' 
#' - **`causal_targets_layers_lst()`**: Extract layers for specific target words or 
#'   phrases based on their contexts. Returns hidden states for each target token.
#' - **`causal_words_layers_lst()`**: Extract layers for all words in grouped text
#'   (e.g., sentences or paragraphs).
#' - **`causal_tokens_layers()`**: Extract layers for all tokens in a single text.
#'
#' @param contexts A character vector of context strings (for `causal_targets_layers_lst`).
#' @param targets A character vector of target words/phrases (for `causal_targets_layers_lst`).
#' @param x A character vector of words (for `causal_words_layers_lst`) or a single 
#'   string (for `causal_tokens_layers`).
#' @param by A vector for grouping elements of `x` (for `causal_words_layers_lst`).
#' @param text A single string (for `causal_tokens_layers`).
#' @param layers Integer vector specifying which layers to extract. Use `NULL` 
#'   (default) to extract all layers. Layer 0 is the non-contextualized embeddings.
#' @param return_type Either "list" (default) or "array". If "list", returns a 
#'   named list with one element per layer. If "array", returns a 3D array with
#'   dimensions [layers, tokens, hidden_size].
#' @inheritParams causal_targets_pred
#' @inheritParams causal_words_pred
#' @param merge_fun Function to merge multi-token words into single representations.
#'   Default is `colMeans` which averages across tokens for each dimension. 
#'   The function should take a matrix [n_tokens, hidden_dim] and return 
#'   a vector of length hidden_dim.
#'   Other options: `function(x) x[1,]` (use first token only), 
#'   `colSums` (sum across tokens), or any custom function.
#'   Set to `NULL` to disable merging and return all tokens separately 
#'   (output will be a matrix [n_tokens, hidden_dim]).
#'
#' @return A named list or 3D array of hidden states. Structure depends on 
#'   `return_type`:
#'   - If "list": Named list where each element is a matrix [tokens × hidden_size]
#'   - If "array": 3D array [layers × tokens × hidden_size]
#'   
#'   For `causal_words_layers_lst`, returns a list of such structures (one per group).
#'
#' @examples
#' \dontrun{
#' 
#' # Extract layers for specific targets
#' layers <- causal_targets_layers_lst(
#'   contexts = c("The cat sat on the", "The dog ran in the"),
#'   targets = c("mat", "park"),
#'   model = "gpt2"
#' )
#' 
#' # Extract only layer 0 (embeddings) and layer 12 (final layer)
#' layers_subset <- causal_targets_layers_lst(
#'   contexts = "The apple fell from the",
#'   targets = "tree",
#'   layers = c(0, 12),
#'   model = "gpt2"
#' )
#' 
#' # Get as array instead of list
#' layers_array <- causal_targets_layers_lst(
#'   contexts = "Once upon a",
#'   targets = "time",
#'   return_type = "array",
#'   model = "gpt2"
#' )
#' dim(layers_array)  # [n_layers, n_tokens, hidden_size]
#' }
#'
#' @family causal model functions
#' @export
#' @rdname causal_layer_extraction
causal_targets_layers_lst <- function(contexts,
                                  targets,
                                  sep = " ",
                                  layers = NULL,
                                  include_embeddings = TRUE,
                                  merge_fun = colMeans,  # Default: average
                                  return_type = c("list", "array"),
                                  model = getOption("pangoling.causal.default"),
                                  checkpoint = NULL,
                                  add_special_tokens = NULL,
                                  config_model = NULL,
                                  config_tokenizer = NULL,
                                  batch_size = 1) {
  return_type <- match.arg(return_type)
  
  if(any(!is_really_string(targets))) { 
    stop2("`targets` needs to be a vector of non-empty strings.")
  }
  if(any(!is_really_string(contexts))) {
    stop2("`contexts` needs to be a vector of non-empty strings.")
  }
  
  message_verbose_model(model, checkpoint)
  
  # Load model and tokenizer
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "causal",
                    output_hidden_states = TRUE,
                    config_model = config_model)
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  
  # Combine contexts and targets
  x <- c(rbind(contexts, targets))
  by <- rep(seq_len(length(x)/2), each = 2)
  word_by_word_texts <- split(x, by, drop = TRUE)
  
  pasted_texts <- conc_words(word_by_word_texts, sep = sep)
  
  # Create tensors
  tensors <- create_tensor_lst(
    texts = unname(pasted_texts),
    tkzr = tkzr,
    add_special_tokens = add_special_tokens,
    stride = 1,
    batch_size = batch_size
  )
  
  # Process each tensor to extract layers for target tokens
  result <- lapply(seq_along(tensors), function(i) {
    tensor <- tensors[[i]]
    words <- word_by_word_texts[[i]]
    target <- words[2]  # Second element is always the target
    
    # Get model output with hidden states
    output <- trf(tensor$input_ids)
    
    # Tokenize to find target positions
    context_ids <- get_id(words[1], model = model, 
                          add_special_tokens = add_special_tokens,
                          config_tokenizer = config_tokenizer)[[1]]
    target_ids <- get_id(paste0(sep, target), model = model,
                         add_special_tokens = add_special_tokens, 
                         config_tokenizer = config_tokenizer)[[1]]
    
    # Target starts after context
    n_context_tokens <- length(context_ids)
    target_positions <- seq(n_context_tokens, 
                            n_context_tokens + length(target_ids) - 1)
    
    # Extract hidden states for target positions
    hidden_states <- extract_hidden_states(
      model_output = output,
      layers = layers,
      token_positions = target_positions,
      include_embeddings = include_embeddings,
      model = trf,
      input_ids = tensor$input_ids,
      task = "causal"
    )
    
    # Merge multi-token words if requested
    if (!is.null(merge_fun) && length(target_ids) > 1) {
      # All target tokens should be merged into one word
      token_groups <- list(seq_along(target_positions))
      hidden_states <- merge_tokens(hidden_states, token_groups, merge_fun)
    }
    
    # If merge_fun is NULL and we have multiple tokens, 
    # the output will be [n_tokens, hidden_dim] per layer
    
    # Determine actual layers extracted
    actual_layers <- if (is.null(layers)) {
      if (include_embeddings) {
        c(-1, seq(0, length(output$hidden_states) - 1))
      } else {
        seq(0, length(output$hidden_states) - 1)
      }
    } else {
      layers
    }
    
    format_layer_output(hidden_states, actual_layers, return_type)
  })
  
  names(result) <- targets
  
  # If only one target, return the structure directly
  if (length(result) == 1) {
    return(result[[1]])
  }
  
  result
}

#' @export
#' @rdname causal_layer_extraction
causal_words_layers_lst <- function(x,
                                by = rep(1, length(x)),
                                sep = " ",
                                layers = NULL,
                                include_embeddings = TRUE,
                                merge_fun = colMeans,
                                return_type = c("list", "array"),
                                sorted = FALSE,
                                model = getOption("pangoling.causal.default"),
                                checkpoint = NULL,
                                add_special_tokens = NULL,
                                config_model = NULL,
                                config_tokenizer = NULL,
                                batch_size = 1) {
  return_type <- match.arg(return_type)
  
  # Better error checking
  if (is.data.frame(x)) {
    stop2("`x` must be a character vector, not a data frame. ",
          "Did you mean to use `x = your_df$word_column`?")
  }
  
  if (is.data.frame(by)) {
    stop2("`by` must be a vector, not a data frame. ",
          "Did you mean to use `by = your_df$group_column`?")
  }
  
  if (!is.character(x)) {
    stop2("`x` must be a character vector, got ", class(x)[1], " instead.")
  }
  
  if (any(!is_really_string(x))) {
    stop2("`x` needs to be a vector of non-empty strings.")
  }
  
  if (length(x) != length(by)) {
    stop2("`x` and `by` must have the same length. ",
          "`x` has length ", length(x), " but `by` has length ", length(by), ".")
  }
  
  message_verbose_model(model, checkpoint)
  
  # Load model and tokenizer
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "causal",
                    output_hidden_states = TRUE,
                    config_model = config_model)
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  
  # Group words
  word_by_word_texts <- split(x, by)
  pasted_texts <- conc_words(word_by_word_texts, sep = sep)
  
  # Create tensors
  tensors <- create_tensor_lst(
    texts = unname(pasted_texts),
    tkzr = tkzr,
    add_special_tokens = add_special_tokens,
    stride = 1,
    batch_size = batch_size
  )
  
  # Extract layers for each group
  result <- tidytable::map2(
    tensors,
    word_by_word_texts,
    function(tensor, words) {
      output <- trf(tensor$input_ids)
      
      # Extract all hidden states
      hidden_states <- extract_hidden_states(
        model_output = output,
        layers = layers,
        token_positions = NULL,  # Get all positions
        include_embeddings = include_embeddings,
        model = trf,
        input_ids = tensor$input_ids,
        task = "causal"
      )
      
      # Tokenize each word to get token groups
      word_token_lists <- lapply(seq_along(words), function(i) {
        w <- words[i]
        if (i == 1) {
          get_id(w, model = model,
                 add_special_tokens = add_special_tokens,
                 config_tokenizer = config_tokenizer)[[1]]
        } else {
          get_id(paste0(sep, w), model = model,
                 add_special_tokens = add_special_tokens,
                 config_tokenizer = config_tokenizer)[[1]]
        }
      })
      
      # Create token groups (which tokens belong to which word)
      token_groups <- list()
      current_pos <- 1
      for (i in seq_along(word_token_lists)) {
        n_tokens <- length(word_token_lists[[i]])
        token_groups[[i]] <- seq(current_pos, current_pos + n_tokens - 1)
        current_pos <- current_pos + n_tokens
      }
      
      # Extract layers for each word separately
      word_layers <- lapply(seq_along(words), function(i) {
        word_token_positions <- token_groups[[i]]
        
        # Extract just this word's tokens from each layer
        word_hidden_states <- lapply(hidden_states, function(layer_matrix) {
          layer_matrix[word_token_positions, , drop = FALSE]
        })
        
        # Merge if requested
        if (!is.null(merge_fun)) {
          word_hidden_states <- merge_tokens(
            word_hidden_states, 
            list(seq_along(word_token_positions)),  # All tokens in this word
            merge_fun
          )
          # merge_tokens returns vectors for single words
          # Ensure they stay as vectors (not [1, 768] matrices)
          word_hidden_states <- lapply(word_hidden_states, function(layer_data) {
            if (is.matrix(layer_data) && nrow(layer_data) == 1) {
              # Convert [1, 768] matrix to vector of length 768
              as.vector(layer_data)
            } else {
              layer_data
            }
          })
        }
        
        # Determine actual layers
        actual_layers <- if (is.null(layers)) {
          if (include_embeddings) {
            c(-1, seq(0, length(output$hidden_states) - 1))
          } else {
            seq(0, length(output$hidden_states) - 1)
          }
        } else {
          layers
        }
        
        format_layer_output(word_hidden_states, actual_layers, return_type)
      })
      
      names(word_layers) <- words
      word_layers
    }
  )
  
  names(result) <- levels(as.factor(by))
  if(!sorted) result <- result[unique(as.factor(by))]
  
  result
}


#' @export
#' @rdname causal_layer_extraction
causal_tokens_layers <- function(text,
                                 layers = NULL,
                                 include_embeddings = TRUE,
                                 merge_fun = NULL,
                                 return_type = c("list", "array"),
                                 model = getOption("pangoling.causal.default"),
                                 checkpoint = NULL,
                                 add_special_tokens = NULL,
                                 config_model = NULL,
                                 config_tokenizer = NULL) {
  return_type <- match.arg(return_type)
  
  # Allow multiple texts
  if (!is.character(text)) {
    stop2("`text` must be a character vector, got ", class(text)[1], " instead.")
  }
  
  if (any(!is_really_string(text))) {
    stop2("`text` needs to be a vector of non-empty strings.")
  }
  
  message_verbose_model(model, checkpoint)
  
  # Load model and tokenizer
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "causal",
                    output_hidden_states = TRUE,
                    config_model = config_model)
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  
  # Process each text
  result <- lapply(seq_along(text), function(text_idx) {
    current_text <- text[text_idx]
    
    # Tokenize
    tensor <- encode(list(current_text), tkzr, 
                     add_special_tokens = add_special_tokens)
    
    # Get tokens as strings
    tokens <- tokenize_lst(current_text, 
                           model = model,
                           add_special_tokens = add_special_tokens,
                           config_tokenizer = config_tokenizer)[[1]]
    
    # Get model output
    output <- trf(tensor$input_ids)
    
    # Extract hidden states (all tokens)
    hidden_states <- extract_hidden_states(
      model_output = output,
      layers = layers,
      token_positions = NULL,
      include_embeddings = include_embeddings,
      model = trf,
      input_ids = tensor$input_ids,
      task = "causal"
    )
    
    # Determine actual layers
    actual_layers <- if (is.null(layers)) {
      if (include_embeddings) {
        c(-1, seq(0, length(output$hidden_states) - 1))
      } else {
        seq(0, length(output$hidden_states) - 1)
      }
    } else {
      layers
    }
    
    # If merge_fun is provided, we can't really "merge" at token level
    # because each token is independent. Ignore merge_fun for tokens_layers.
    if (!is.null(merge_fun)) {
      warning("merge_fun is not applicable for causal_tokens_layers and will be ignored.")
    }
    
    # Restructure: instead of [layers -> tokens x hidden_dim]
    # Return: [tokens -> layers -> hidden_dim vector]
    n_tokens <- length(tokens)
    
    token_layers <- lapply(seq_len(n_tokens), function(token_idx) {
      # Extract this token from all layers
      token_hidden_states <- lapply(hidden_states, function(layer_matrix) {
        as.vector(layer_matrix[token_idx, ])
      })
      
      format_layer_output(token_hidden_states, actual_layers, return_type)
    })
    
    names(token_layers) <- tokens
    token_layers
  })
  
  # If only one text, return the structure directly (not wrapped in list)
  if (length(result) == 1) {
    return(result[[1]])
  }
  
  # For multiple texts, name them
  if (!is.null(names(text))) {
    names(result) <- names(text)
  } else {
    names(result) <- paste0("text_", seq_along(text))
  }
  
  result
}
