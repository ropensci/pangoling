   contexts = c("The cat sat on the", "The dog ran in the")
   targets = c("mat", "park")
   model = "gpt2"

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

                                  sep = " "
                                  layers = NULL
                                  include_embeddings = TRUE
                                  merge_fun = colMeans
                                  return_type = c("list", "array")

                                  checkpoint = NULL
                                  add_special_tokens = NULL
                                  config_model = NULL
                                  config_tokenizer = NULL
                                  batch_size = 1


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
