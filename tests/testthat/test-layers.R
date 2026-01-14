options(pangoling.verbose = FALSE)


test_that("causal_targets_layers returns correct structure as list", {
  skip_if_not(installed_py_pangoling())
  
  causal_unload()

  result <- causal_targets_layers(
    contexts = "The cat sat on the",
    targets = "mat",
    model = "gpt2",
    return_type = "list"
  )
  
  # Should be a named list
  expect_type(result, "list")
  expect_true(all(grepl("^layer_", names(result))))
  
  # Each element should be a matrix
  expect_true(all(sapply(result, is.matrix)))
  
  # All matrices should have same hidden dimension (768 for GPT-2)
  hidden_dims <- sapply(result, ncol)
  expect_true(all(hidden_dims == 768))
})

test_that("causal_targets_layers returns correct structure as array", {
  skip_if_not(installed_py_pangoling())
  
  result <- causal_targets_layers(
    contexts = "The cat sat on the",
    targets = "mat",
    model = "gpt2",
    return_type = "array"
  )
  
  # Should be a 3D array
  expect_true(is.array(result))
  expect_equal(length(dim(result)), 3)
  
  # Dimensions: [layers, tokens, hidden_size]
  # GPT-2 has 13 layers (0-12), hidden_size is 768
  expect_equal(dim(result)[3], 13)  # 13 layers
  expect_equal(dim(result)[2], 768) # hidden dimension
})

test_that("layer selection works correctly", {
  skip_if_not(installed_py_pangoling())
  
  # Select only layers 0 and 12
  result <- causal_targets_layers(
    contexts = "The cat",
    targets = "sat",
    layers = c(0, 12),
    model = "gpt2",
    return_type = "list"
  )
  
  # Should have exactly 2 layers
  expect_equal(length(result), 2)
  expect_equal(names(result), c("layer_0", "layer_12"))
})

test_that("layer 0 contains non-contextualized embeddings", {
  skip_if_not(installed_py_pangoling())
  
  # Extract layer 0 for same token in different contexts
  result1 <- causal_targets_layers(
    contexts = "The cat",
    targets = "sat",
    layers = 0,
    model = "gpt2",
    return_type = "list"
  )
  
  result2 <- causal_targets_layers(
    contexts = "Yesterday I",
    targets = "sat",
    layers = 0,
    model = "gpt2",
    return_type = "list"
  )
  
  # Layer 0 embeddings should be identical for same token regardless of context
  # (assuming single-token word)
    expect_equal(result1$layer_0, result2$layer_0, tolerance = 1e-6)
})

test_that("higher layers show contextualization", {
  skip_if_not(installed_py_pangoling())
  
  # Extract final layer for same token in different contexts
  result1 <- causal_targets_layers(
    contexts = "The cat",
    targets = "sat",
    layers = 12,
    model = "gpt2",
    return_type = "list"
  )
  
  result2 <- causal_targets_layers(
    contexts = "Yesterday I",
    targets = "sat",
    layers = 12,
    model = "gpt2",
    return_type = "list"
  )
  
  # Layer 12 embeddings should differ due to contextualization
  # (assuming single-token word)
    # Using correlation as they might be similar but not identical
    correlation <- cor(as.vector(result1$layer_12), as.vector(result2$layer_12))
    expect_lt(correlation, 0.99)  # Should be different

})

test_that("causal_words_layers_lst groups correctly", {
  skip_if_not(installed_py_pangoling())
  
  result <- causal_words_layers_lst(
    x = c("The", "cat", "sat", "The", "dog", "ran"),
    by = c(1, 1, 1, 2, 2, 2),
    model = "gpt2",
    layers = c(0, 12),
    return_type = "list"
  )
  
  # Should return a list with 2 groups
  expect_equal(length(result), 2)
  expect_equal(names(result), c("1", "2"))
  
  # Each group should have 2 layers
  expect_equal(length(result[[1]]), 2)
  expect_equal(length(result[[2]]), 2)
})

test_that("causal_tokens_layers works for single text", {
  skip_if_not(installed_py_pangoling())
  
  result <- causal_tokens_layers(
    text = "The cat sat on the mat",
    layers = c(0, 6, 12),
    model = "gpt2",
    return_type = "list"
  )
  
  # Should have 3 layers
  expect_equal(length(result), 3)
  expect_equal(names(result), c("layer_0", "layer_6", "layer_12"))
  
  # Each layer should be a matrix
  expect_true(all(sapply(result, is.matrix)))
})

# test_that("masked_targets_layers works with bidirectional context", {
#   skip_if_not(installed_py_pangoling())
#   
#   result <- masked_targets_layers(
#     prev_contexts = "The cat sat on the",
#     targets = "mat",
#     after_contexts = "near the door",
#     layers = c(0, 12),
#     model = "bert-base-uncased",
#     return_type = "list"
#   )
#   
#   # Should have 2 layers
#   expect_equal(length(result), 2)
#   expect_equal(names(result), c("layer_0", "layer_12"))
#   
#   # Each should be a matrix
#   expect_true(all(sapply(result, is.matrix)))
# })


test_that("error handling for invalid layer indices", {
  skip_if_not(installed_py_pangoling())
  
  # GPT-2 has layers 0-12, so 13 is invalid
  expect_error(
    causal_targets_layers(
      contexts = "The cat",
      targets = "sat",
      layers = c(0, 13),
      model = "gpt2"
    ),
    "Layer indices must be between"
  )
  
  # Negative indices should error
  expect_error(
    causal_targets_layers(
      contexts = "The cat",
      targets = "sat",
      layers = c(-1, 0),
      model = "gpt2"
    ),
    "Layer indices must be between"
  )
})

test_that("multiple targets return named list", {
  skip_if_not(installed_py_pangoling())
  
  result <- causal_targets_layers(
    contexts = c("The cat", "The dog"),
    targets = c("sat", "ran"),
    layers = c(0, 12),
    model = "gpt2"
  )
  
  # Should be a list with target names
  expect_type(result, "list")
  expect_equal(names(result), c("sat", "ran"))
  
  # Each target should have layer data
  expect_equal(length(result$sat), 2)
  expect_equal(length(result$ran), 2)
})

test_that("token positions are correctly identified", {
  skip_if_not(installed_py_pangoling())
  
  # Test with multi-token word
  result <- causal_targets_layers(
    contexts = "I love eating",
    targets = "strawberries.",  # Multi-token word
    layers = 0,
    model = "gpt2",
    return_type = "list"
  )
  
  # Should extract two tokens of "strawberries."
  # GPT-2 tokenizes "strawberries." into 2 tokens
  expect_true(nrow(result$layer_0) == 2)
})


test_that("causal_targets_layers, causal_words_layers_lst, and causal_tokens_layers produce consistent results", {
  skip_if_not(installed_py_pangoling())
  
  causal_preload(model = "gpt2", output_hidden_states = TRUE)
  
  # Test text
  words <- c("The", "cat", "sat")
  text <- paste(words, collapse = " ")
  
  # Extract using tokens_layers (entire text)
  layers_tokens <- causal_tokens_layers(
    text = text,
    layers = c(-1, 0, 12),
    merge_fun = NULL,  # Keep all tokens
    model = "gpt2"
  )
  
  # Extract using words_layers (grouped by sentence)
  layers_words <- causal_words_layers_lst(
    x = words,
    by = rep(1, length(words)),
    layers = c(-1, 0, 12),
    merge_fun = NULL,  # Keep all tokens
    model = "gpt2"
  )
  
  # They should match (words_layers returns list, so extract first element)
  expect_equal(layers_tokens$token_embeddings, 
               layers_words[[1]]$token_embeddings,
               tolerance = 1e-5)
  
  expect_equal(layers_tokens$layer_0, 
               layers_words[[1]]$layer_0,
               tolerance = 1e-5)
  
  expect_equal(layers_tokens$layer_12, 
               layers_words[[1]]$layer_12,
               tolerance = 1e-5)
})

test_that("causal_targets_layers with merged tokens matches expected dimensions", {
  skip_if_not(installed_py_pangoling())
  
  # Test with multi-token word
  layers_merged <- causal_targets_layers(
    contexts = "I love eating",
    targets = "strawberries.",
    layers = c(-1, 0, 12),
    merge_fun = colMeans,  # Merge tokens
    model = "gpt2"
  )
  
  layers_unmerged <- causal_targets_layers(
    contexts = "I love eating",
    targets = "strawberries.",
    layers = c(-1, 0, 12),
    merge_fun = NULL,  # Don't merge
    model = "gpt2"
  )
  
  # Merged should be [1, 768]
  expect_equal(length(layers_merged$layer_0), 768)

  # Unmerged should be [n_tokens, 768] where n_tokens > 1
  expect_gt(nrow(layers_unmerged$layer_0), 1)
  expect_equal(ncol(layers_unmerged$layer_0), 768)
  
  # Manual merge should match automatic merge
  manual_merge <- colMeans(layers_unmerged$layer_0)
  expect_equal(as.vector(layers_merged$layer_0), 
               manual_merge,
               tolerance = 1e-5)
})



test_python_equivalence(
  text = "The cat sat on the mat",
  token_position = 6,  # Position of "mat" (R 1-indexed)
  model = "gpt2"
)
