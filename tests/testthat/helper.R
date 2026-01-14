# R/test-helpers.R

#' Expect R layer extraction to match Python
#' @keywords internal
expect_python_equivalence <- function(text = "The cat sat on the mat",
                                      token_position = 6,
                                      model = "gpt2",
                                      layers = c(-1, 0, 12),
                                      tolerance = 1e-5) {
  
  if (!installed_py_pangoling()) {
    testthat::skip("Python pangoling dependencies not available")
  }
  
  # Preload R model
  causal_preload(model = model, output_hidden_states = TRUE)
  
  # R extraction
  layers_r <- causal_tokens_layers(
    text = text,
    layers = layers,
    model = model
  )
  
  # Python extraction
  py_position <- token_position - 1
  
  # Build Python code
  layer_extractions <- paste(
    sapply(layers[layers >= 0], function(l) {
      sprintf("layer_%d = hidden_states[%d][0, %d, :].detach().numpy()", 
              l, l, py_position)
    }),
    collapse = "\n"
  )
  
  py_code <- sprintf("
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('%s')
tokenizer = GPT2Tokenizer.from_pretrained('%s')
model.eval()

text = '%s'
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

token_embed = model.transformer.wte(inputs['input_ids'])[0, %d, :].detach().numpy()
%s
", model, model, text, py_position, layer_extractions)
  
  reticulate::py_run_string(py_code)
  
  # Compare each requested layer using expect_equal directly
  results <- list()
  
  if (-1 %in% layers) {
    token_embed_py <- reticulate::py$token_embed
    testthat::expect_equal(
      as.vector(layers_r$token_embeddings[token_position, ]),
      as.vector(token_embed_py),
      tolerance = tolerance,
      label = "R token_embeddings",
      expected.label = "Python token_embeddings"
    )
    results$token_embeddings <- TRUE
  }
  
  for (layer_idx in layers[layers >= 0]) {
    layer_name <- paste0("layer_", layer_idx)
    layer_py <- reticulate::py[[layer_name]]
    
    testthat::expect_equal(
      as.vector(layers_r[[layer_name]][token_position, ]),
      as.vector(layer_py),
      tolerance = tolerance,
      label = paste0("R ", layer_name),
      expected.label = paste0("Python ", layer_name)
    )
    results[[layer_name]] <- TRUE
  }
  
  invisible(results)
}

#' Expect layer extraction functions to produce consistent results
#' @keywords internal
expect_layer_consistency <- function(words = c("The", "cat", "sat"),
                                     model = "gpt2",
                                     layers = c(-1, 0, 12),
                                     merge_fun = NULL,
                                     tolerance = 1e-5) {
  
  if (!installed_py_pangoling()) {
    testthat::skip("Python pangoling dependencies not available")
  }
  
  causal_preload(model = model, output_hidden_states = TRUE)
  
  text <- paste(words, collapse = " ")
  
  # Extract with tokens_layers
  layers_tokens <- causal_tokens_layers(
    text = text,
    layers = layers,
    model = model
  )
  
  # Extract with words_layers
  layers_words <- causal_words_layers_lst(
    x = words,
    by = rep(1, length(words)),
    layers = layers,
    merge_fun = merge_fun,
    model = model
  )
  
  results <- list()
  
  if (is.null(merge_fun)) {
    # No merging - compare token-level
    token_idx <- 1
    for (word in words) {
      word_layers <- layers_words[[1]][[word]]
      
      # Get the first layer name to determine number of tokens
      first_layer <- word_layers[[1]]
      n_word_tokens <- if (is.matrix(first_layer)) nrow(first_layer) else 1
      
      for (layer_name in names(layers_tokens)) {
        token_range <- seq(token_idx, token_idx + n_word_tokens - 1)
        tokens_extract <- layers_tokens[[layer_name]][token_range, , drop = FALSE]
        
        testthat::expect_equal(
          tokens_extract,
          word_layers[[layer_name]],
          tolerance = tolerance,
          label = paste0("tokens_layers[", word, "]$", layer_name),
          expected.label = paste0("words_layers$", word, "$", layer_name)
        )
        results[[paste0(word, "_", layer_name)]] <- TRUE
      }
      
      token_idx <- token_idx + n_word_tokens
    }
  } else {
    # Merging - each word should be a vector
    for (word in words) {
      word_layers <- layers_words[[1]][[word]]
      
      for (layer_name in names(word_layers)) {
        # Check it's a vector of length 768
        testthat::expect_null(
          dim(word_layers[[layer_name]]),
          info = paste("Merged word", word, layer_name, "should be a vector")
        )
        testthat::expect_equal(
          length(word_layers[[layer_name]]), 
          768,
          info = paste("Merged word", word, layer_name, "should have length 768")
        )
        results[[paste0(word, "_", layer_name, "_is_vector")]] <- TRUE
      }
    }
  }
  
  invisible(results)
}

#' Expect targets_layers and words_layers to match
#' @keywords internal
expect_targets_words_match <- function(words = c("The", "cat", "sat"),
                                       target_index = length(words),
                                       model = "gpt2",
                                       layers = c(-1, 0, 12),
                                       merge_fun = colMeans,
                                       tolerance = 1e-5) {
  
  if (!installed_py_pangoling()) {
    testthat::skip("Python pangoling dependencies not available")
  }
  
  causal_preload(model = model, output_hidden_states = TRUE)
  
  target_word <- words[target_index]
  context_words <- if (target_index > 1) words[1:(target_index - 1)] else character(0)
  context <- if (length(context_words) > 0) paste(context_words, collapse = " ") else ""
  
  # Extract with words_layers
  w_layers <- causal_words_layers_lst(
    x = words,
    by = rep(1, length(words)),
    layers = layers,
    merge_fun = merge_fun,
    model = model
  )
  
  # Extract with targets_layers
  if (context == "") {
    t_layers <- causal_words_targets_lst(
      contexts = "",
      targets = target_word,
      layers = layers,
      merge_fun = merge_fun,
      model = model
    )
  } else {
    t_layers <- causal_words_targets_lst(
      contexts = context,
      targets = target_word,
      layers = layers,
      merge_fun = merge_fun,
      model = model
    )
  }
  
  results <- list()
  
  # Compare each layer
  for (layer_name in names(t_layers)) {
    testthat::expect_equal(
      w_layers[[1]][[target_word]][[layer_name]],
      t_layers[[layer_name]],
      tolerance = tolerance,
      label = paste0("words_layers$", target_word, "$", layer_name),
      expected.label = paste0("targets_layers$", layer_name)
    )
    results[[layer_name]] <- TRUE
  }
  
  invisible(results)
}