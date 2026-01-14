test_that("R layer extraction matches Python", {
  expect_python_equivalence(
    text = "The cat sat on the mat",
    token_position = 6,
    model = "gpt2",
    layers = c(-1, 0, 12)
  )
})

test_that("R matches Python for different texts", {
  expect_python_equivalence(
    text = "Hello world",
    token_position = 2,
    model = "gpt2"
  )
  
  expect_python_equivalence(
    text = "I love strawberries",
    token_position = 3,
    model = "gpt2"
  )
})

test_that("tokens_layers and words_layers are consistent (unmerged)", {
  expect_layer_consistency(
    words = c("The", "cat", "sat"),
    model = "gpt2",
    merge_fun = NULL  # No merging
  )
})

test_that("tokens_layers and words_layers are consistent (merged)", {
  expect_layer_consistency(
    words = c("The", "cat", "sat"),
    model = "gpt2",
    merge_fun = colMeans
  )
})

test_that("targets_layers and words_layers match for single words", {
  expect_targets_words_match(
    words = c("The", "cat", "sat"),
    target_index = 2,  # Test "cat"
    model = "gpt2",
    merge_fun = colMeans
  )
  
  expect_targets_words_match(
    words = c("The", "cat", "sat"),
    target_index = 3,  # Test "sat"
    model = "gpt2",
    merge_fun = colMeans
  )
})

test_that("targets_layers and words_layers match for multi-token words", {
  expect_targets_words_match(
    words = c("I", "love", "strawberries"),
    target_index = 3,  # Multi-token word
    model = "gpt2",
    merge_fun = colMeans
  )
  
  expect_targets_words_match(
    words = c("I", "love", "strawberries"),
    target_index = 3,
    model = "gpt2",
    merge_fun = NULL  # Unmerged
  )
})

test_that("layer consistency with different layer selections", {
  expect_layer_consistency(
    words = c("The", "quick", "fox"),
    model = "gpt2",
    layers = c(0, 6, 12),  # Skip embeddings
    merge_fun = colMeans
  )
  
  expect_layer_consistency(
    words = c("The", "quick", "fox"),
    model = "gpt2",
    layers = c(-1),  # Only embeddings
    merge_fun = colMeans
  )
})