#' Preloads a masked language model
#'
#' Preloads a masked language model to speed up next runs.
#'
#' A masked language model (also called BERT-like, or encoder model) is a type
#'  of large language model  that can be used to predict the content of a mask
#'  in a sentence.
#'
#' If not specified, the masked model that will be used is the one set in
#' specified in the global option `pangoling.masked.default`, this can be
#' accessed via `getOption("pangoling.masked.default")` (by default
#' "`r getOption("pangoling.masked.default")`"). To change the default option
#'  use `options(pangoling.masked.default = "newmaskedmodel")`.
#'
#' A list of possible masked can be found in
#' [Hugging Face website](https://huggingface.co/models?pipeline_tag=fill-mask)
#'
#' Using the  `config_model` and `config_tokenizer` arguments, it's possible to
#'  control how the model and tokenizer from Hugging Face is accessed, see the
#'  python method
#'  [`from_pretrained`](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained)
#' for details. In case of errors check the status of
#'  [https://status.huggingface.co/](https://status.huggingface.co/)
#'
#' @inheritParams causal_preload
#' @param model Name of a pre-trained model or folder. One should be able to use
#' models based on "bert". See 
#' [hugging face website](https://huggingface.co/models?other=bert).
#' @return Nothing.
#'
#' @examplesIf installed_py_pangoling()
#' causal_preload(model = "bert-base-uncased")
#'
#' @family masked model helper functions
#' @export
#'
masked_preload <- function(model = getOption("pangoling.masked.default"),
                           output_hidden_states = FALSE,
                           add_special_tokens = NULL,
                           config_model = NULL, config_tokenizer = NULL) {
  message_verbose("Preloading masked model ", model, "...")

  lang_model(model, 
             task = "masked", 
             output_hidden_states = output_hidden_states,
             config_model = config_model)
  tokenizer(model, add_special_tokens = add_special_tokens, config_tokenizer)
  invisible()
}

#' Unload a masked language model and free memory
#'
#' Unloads a masked language model and its tokenizer from memory, and triggers
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
#' masked_preload(model = "bert-base-uncased")
#' 
#' # Do some work...
#' pred <- masked_targets_pred(
#'   prev_contexts = "The cat", 
#'   targets = "sat",
#'   after_contexts = "down",
#'   model = "bert-base-uncased"
#' )
#' 
#' # Unload when done
#' masked_unload()
#'
#' @family masked model helper functions
#' @export
masked_unload <- function() {
  transformer_unload()
}
#' Returns the configuration of a masked model
#'
#' Returns the configuration of a masked model.
#'
#' @inheritParams masked_preload
#' @inherit  masked_preload details
#' @return A list with the configuration of the model.
#' @examplesIf installed_py_pangoling()
#' masked_config(model = "bert-base-uncased")
#'
#' @family masked model helper functions
#' @export
masked_config <- function(model = getOption("pangoling.masked.default"),
                          config_model = NULL) {
  lang_model(
    model = model,
    task = "masked",
    config_model = config_model
  )$config$to_dict()
}

#' Get the possible tokens and their log probabilities for each mask in a 
#'  sentence
#'
#' For each mask, indicated with `[MASK]`, in a sentence, get the possible 
#' tokens and their  predictability (by default the natural logarithm of the 
#' word probability) using a masked transformer.
#'
#' @section More examples:
#' See the
#' [online article](https://docs.ropensci.org/pangoling/articles/intro-bert.html)
#' in pangoling website for more examples.
#'
#'
#' @param masked_sentences Masked sentences.
#' @inheritParams masked_preload
#' @inheritParams causal_words_pred
#' @inherit masked_preload details
#' @return A table with the masked sentences, the tokens (`token`),
#'         predictability (`pred`), and the respective mask number (`mask_n`).
#'
#' @examplesIf installed_py_pangoling()
#' masked_tokens_pred_tbl("The [MASK] doesn't fall far from the tree.",
#'   model = "bert-base-uncased"
#' )
#'
#' @family masked model functions
#' @export
masked_tokens_pred_tbl <- 
  function(masked_sentences,
           log.p = getOption("pangoling.log.p"),
           model = getOption("pangoling.masked.default"),
           checkpoint = NULL,
           add_special_tokens = NULL,
           config_model = NULL,
           config_tokenizer = NULL) {
    message_verbose_model(model, checkpoint = checkpoint, causal = FALSE)
    tkzr <- tokenizer(model,
                      add_special_tokens = add_special_tokens,
                      config_tokenizer = config_tokenizer
                      )
    trf <- lang_model(model,
                      checkpoint = checkpoint,
                      task = "masked",
                      config_model = config_model
                      )
    vocab <- get_vocab(tkzr)
    # non_batched:
    # TODO: speedup using batches
    tidytable::map_dfr(masked_sentences, function(masked_sentence) {
      masked_tensor <- encode(list(masked_sentence), tkzr,
                              add_special_tokens = add_special_tokens
                              )$input_ids
      outputs <- trf(masked_tensor)
      mask_pos <- which(masked_tensor$tolist()[[1]] == tkzr$mask_token_id)
      logits_masks <- outputs$logits[0][mask_pos - 1] # python starts in 0
      lp <- reticulate::py_to_r(
                          torch$log_softmax(logits_masks, dim = -1L)$tolist()
                        )
      if (length(mask_pos) <= 1) lp <- list(lp) # to keep it consistent
      # names(lp) <-  1:length(lp)
      if (length(mask_pos) == 0) {
        tidytable::tidytable(
                     masked_sentence = masked_sentence,
                     token = NA,
                     pred = NA,
                     mask_n = NA
                   )
      } else {
        lp |> 
          tidytable::map_dfr(~
                               tidytable::tidytable(
                                            masked_sentence = masked_sentence,
                                            token = vocab,
                                            pred = ln_p_change(.x,
                                                               log.p = log.p)
                                          ) |>
                               tidytable::arrange(-pred), .id = "mask_n")
      }
    }) |>
      tidytable::relocate(mask_n, .after = tidyselect::everything())
  }

#' Get the predictability of a target word (or phrase) given a left and right 
#'   context
#'
#' Get the predictability (by default the natural logarithm of the word 
#' probability) of a vector of target words (or phrase) given a
#' vector of left and of right contexts using a masked transformer.
#'
#' @section More examples:
#' See the
#' [online article](https://docs.ropensci.org/pangoling/articles/intro-bert.html)
#' in pangoling website for more examples.
#'
#'
#' @param prev_contexts Left context of the target word in left-to-right written
#'                      languages.
#' @param targets Target words.
#' @param after_contexts Right context of the target in left-to-right written 
#'                       languages.
#' @inheritParams masked_preload
#' @inheritParams causal_words_pred
#' @inherit masked_preload details
#' @return A named vector of predictability values (by default the natural 
#'         logarithm of the word probability).
#' @examplesIf installed_py_pangoling()
#' masked_targets_pred(
#'   prev_contexts = c("The", "The"),
#'   targets = c("apple", "pear"),
#'   after_contexts = c(
#'     "doesn't fall far from the tree.",
#'     "doesn't fall far from the tree."
#'   ),
#'   model = "bert-base-uncased"
#' )
#'
#' @family masked model functions
#' @export
masked_targets_pred <- function(prev_contexts,
                                targets,
                                after_contexts,
                                log.p = getOption("pangoling.log.p"),
                                ignore_regex = "",
                                model = getOption("pangoling.masked.default"),
                                checkpoint = NULL,
                                add_special_tokens = NULL,
                                config_model = NULL,
                                config_tokenizer = NULL) {
  if(any(!is_really_string(targets))) {
    stop2("`targets` needs to be a vector of non-empty strings.")
  }
  stride <- 1
  message_verbose_model(model, checkpoint = checkpoint, causal = FALSE)

  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "masked",
                    config_model = config_model)


  target_tokens <- lapply(targets, tkzr$tokenize)
  masked_sentences <- tidytable::pmap_chr(
                                   list(
                                     prev_contexts,
                                     target_tokens,
                                     after_contexts
                                   ),
                                   function(l, target, r) {
                                     paste0(
                                       l,
                                       " ",
                                       paste0(rep(tkzr$mask_token, 
                                                  length(target)), 
                                              collapse = ""),
                                       " ",
                                       r
                                     )
                                   }
                                 )

  # named tensor list:
  tensors_lst <- tidytable::map2(masked_sentences, targets, function(t, w) {
    l <- create_tensor_lst(t,
                           tkzr,
                           add_special_tokens = add_special_tokens,
                           stride = stride
                           )
    names(l) <- w
    l
  })

  out <- tidytable::pmap(
                      list(targets, prev_contexts, after_contexts, tensors_lst),
                      function(words, l, r, tensor_lst) {
                        # TODO: make it by batches
                        # words <- targets[[1]]
                        # l <- prev_contexts[[1]]
                        # r <- after_contexts[[1]]
                        # tensor_lst <- tensors_lst[[1]]
                        ls_mat <- 
                          masked_lp_mat(tensor_lst =
                                          lapply(tensor_lst,
                                                 function(t) t$input_ids),
                                        trf = trf,
                                        tkzr = tkzr,
                                        add_special_tokens = add_special_tokens,
                                        stride = stride)
                        text <- paste0(words, collapse = " ")
                        tokens <- tkzr$tokenize(text)
                        lapply(ls_mat, function(m) {
                          # m <- ls_mat[[1]]
                          message_verbose(l, " [", words, "] ", r)

                          word_lp(words,
                                  #sep doesn't matter for Bert tokenizer
                                  sep = "",
                                  mat = m,
                                  ignore_regex = ignore_regex,
                                  model = model,
                                  # we don't want to add anything to the target
                                  add_special_tokens = FALSE,
                                  config_tokenizer = config_tokenizer
                                  )
                        })
                      }
                    )

  message_verbose("***\n")
  unlist(out, recursive = TRUE) |>
    ln_p_change(log.p = log.p)

}


#' @noRd
masked_lp_mat <- function(tensor_lst,
                          trf,
                          tkzr,
                          add_special_tokens = NULL,
                          stride = 1,
                          N_pred = 1) {
  tensor <- torch$row_stack(unname(tensor_lst))
  words <- names(tensor_lst)
  tokens <- lapply(words, tkzr$tokenize)
  n_masks <- sum(tensor_lst[[1]]$tolist()[[1]] == tkzr$mask_token_id)
  message_verbose(
    "Processing ",
    tensor$shape[0],
    " batch(es) of ",
    tensor$shape[1],
    " tokens."
  )

  out_lm <- trf(tensor)
  logits_b <- out_lm$logits

  is_masked_lst <- lapply(tensor_lst, function(t) {
    # t <- tensor_lst[[1]]
    id_vector <- t$tolist()[[1]]
    id_vector %in% tkzr$mask_token_id
  })
  # number of predictions ahead
  # if(is.null(N_pred)) N_pred <- sum(is_masked_lst[[1]])
  if (is.null(N_pred)) N_pred <- length(words)

  lmat <- lapply(1:N_pred, function(n_pred) {
    logits_masked <- lapply(seq_along(tensor_lst), function(n) {
      # n <- 1
      # logits is a python object indexed from 0
      if ((n - n_pred) < 0) {
        return(NULL)
      }
      n_masks_here <- length(tokens[[n]])
      n_pred_element <- which(is_masked_lst[[n]])[1:n_masks_here]
      # if(!is_masked_lst[[n]][n_pred_element] #outside of masked elements
      #    || anyNA(n_pred_element)) {
      #   return(NULL)
      # }
      # iterates over sentences
      logits_b[n - n_pred][n_pred_element - 1]
    })
    logits_masked_cleaned <-
      logits_masked[lengths(logits_masked) > 0] |>
      torch$row_stack()
    lp <- reticulate::py_to_r(torch$log_softmax(logits_masked_cleaned,
                                                dim = -1L
                                                ))$tolist()
    mat <- do.call("cbind", lp)
    # columns are not named
    mat_NA <- matrix(NA,
                     nrow = nrow(mat),
                     ncol = sum(lengths(logits_masked) == 0)
                     )
    # add NA columns for predictions not made
    mat <- cbind(mat_NA, mat)
    colnames(mat) <- unlist(tokens)
    rownames(mat) <- get_vocab(tkzr)
    mat
  })
  gc(full = TRUE)
  lmat
}


#' @export
masked_targets_layers <- function(prev_contexts,
                                  targets,
                                  after_contexts,
                                  layers = NULL,
                                  include_embeddings = TRUE,
                                  merge_fun = rowMeans,
                                  return_type = c("list", "array"),
                                  model = getOption("pangoling.masked.default"),
                                  checkpoint = NULL,
                                  add_special_tokens = NULL,
                                  config_model = NULL,
                                  config_tokenizer = NULL) {
  return_type <- match.arg(return_type)
  
  if(any(!is_really_string(targets))) {
    stop2("`targets` needs to be a vector of non-empty strings.")
  }
  
  message_verbose_model(model, checkpoint = checkpoint, causal = FALSE)
  
  # Load model and tokenizer
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "masked",
                    output_hidden_states = TRUE,
                    config_model = config_model)
  
  # Process each target
  result <- tidytable::pmap(
    list(prev_contexts, targets, after_contexts),
    function(prev, target, after) {
      # Create full sentence with target (not masked)
      full_text <- paste(prev, target, after)
      
      # Tokenize to find target positions
      tensor <- encode(list(full_text), tkzr,
                       add_special_tokens = add_special_tokens)
      
      # Get model output
      output <- trf(tensor$input_ids)
      
      # Find target token positions
      prev_ids <- get_id(prev, model = model,
                         add_special_tokens = FALSE,
                         config_tokenizer = config_tokenizer)[[1]]
      target_ids <- get_id(target, model = model,
                           add_special_tokens = FALSE,
                           config_tokenizer = config_tokenizer)[[1]]
      
      # Account for special tokens at start (e.g., [CLS])
      special_start <- if (is.null(add_special_tokens) || add_special_tokens) 1 else 0
      
      target_start <- special_start + length(prev_ids)
      target_positions <- seq(target_start, 
                              target_start + length(target_ids) - 1)
      
      # Extract hidden states
      hidden_states <- extract_hidden_states(
        model_output = output,
        layers = layers,
        token_positions = target_positions,
        include_embeddings = include_embeddings,
        model = trf,
        input_ids = tensor$input_ids,
        task = "masked"
      )
      
      # Merge multi-token words if requested
      if (!is.null(merge_fun) && length(target_ids) > 1) {
        token_groups <- list(seq_along(target_positions))
        hidden_states <- merge_tokens(hidden_states, token_groups, merge_fun)
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
      
      format_layer_output(hidden_states, actual_layers, return_type)
    }
  )
  
  names(result) <- targets
  
  if (length(result) == 1) {
    return(result[[1]])
  }
  
  result
}