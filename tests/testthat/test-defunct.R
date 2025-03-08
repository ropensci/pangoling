options(pangoling.verbose = FALSE)

test_that("Defunct functions return errors", {
  expect_error(
    masked_tokens_tbl("The apple doesn't fall far from the [MASK]"),
    "masked_tokens_pred_tbl"
  )
  
  expect_error(
    masked_lp(l_contexts = "The apple doesn't fall far from the", 
              targets = "tree",
              r_contexts = "."),
    "masked_targets_pred"
  )
  
  expect_error(
    causal_next_tokens_tbl(
      context = "The apple doesn't fall far from the"),
    "causal_next_tokens_pred_tbl"
  )
  
  expect_error(
    causal_lp(x = c("The", "apple", "falls"), by = c(1,1,1)),
    "causal_targets_pred"
  )
  
  expect_error(
    causal_tokens_lp_tbl(
      texts = "The apple doesn't fall far from the tree."),
    "causal_tokens_pred_lst"
  )
  
  expect_error(
    causal_lp_mats(x = c("The", "apple", "falls"), 
                   by = c(1,1,1)),
    "causal_pred_mats"
  )
})
