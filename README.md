
# pangoling <a href="https://docs.ropensci.org/pangoling/"><img src="man/figures/logo.png" align="right" height="139" /></a>

<!-- badges: start -->

[![Codecov test
coverage](https://codecov.io/gh/ropensci/pangoling/branch/main/graph/badge.svg)](https://app.codecov.io/gh/ropensci/pangoling?branch=main)
[![Lifecycle:
stable](https://img.shields.io/badge/lifecycle-stable-green.svg)](https://lifecycle.r-lib.org/articles/stages.html#stable)
[![R-CMD-check](https://github.com/ropensci/pangoling/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/ropensci/pangoling/actions/workflows/R-CMD-check.yaml)
[![Project Status:
active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15008685.svg)](https://doi.org/10.5281/zenodo.15008685)
[![Status at rOpenSci Software Peer
Review](https://badges.ropensci.org/575_status.svg)](https://github.com/ropensci/software-review/issues/575)
[![CRAN
status](https://www.r-pkg.org/badges/version/pangoling)](https://CRAN.R-project.org/package=pangoling)
[![metacran
downloads](https://cranlogs.r-pkg.org/badges/grand-total/pangoling)](https://cran.r-project.org/package=pangoling)

<!-- badges: end -->

`pangoling`[^1] is an R package for estimating the predictability of
words in a given context using transformer models. The package provides
an interface for utilizing pre-trained transformer models (such as GPT-2
or BERT) to obtain word probabilities. These word probabilities are
often utilized as predictors in psycholinguistic studies. This package
can be useful for researchers in the field of psycholinguistics who want
to leverage the power of transformer models in their work.

The package is mostly a wrapper of the python package
[`transformers`](https://pypi.org/project/transformers/) to process data
in a convenient format.

## Important! Limitations and bias

The training data of the most popular models (such as GPT-2) haven’t
been released, so one cannot inspect it. It’s clear that the data
contain a lot of unfiltered content from the internet, which is far from
neutral. See for example the scope in the [openAI team’s model card for
GPT-2](https://github.com/openai/gpt-2/blob/master/model_card.md#out-of-scope-use-cases),
but it should be the same for many other models, and the [limitations
and bias section of GPT-2 in Hugging Face
website](https://huggingface.co/gpt2).

## Installation

To install the latest CRAN version of `pangoling` use:

``` r
install.packages("pangoling")
```

To install the latest version from github use:

``` r
install.packages("pangoling", repos = "https://ropensci.r-universe.dev")
```

`install_py_pangoling` function facilitates the installation of Python
packages needed for using pangoling within an R environment, using the
`reticulate` package for managing Python environments. This needs to be
done once.

``` r
install_py_pangoling()
```

## Example

This is a basic example which shows you how to get log-probabilities of
words in a dataset:

``` r
library(pangoling)
library(tidytable) #fast alternative to dplyr
```

Given a (toy) dataset where sentences are organized with one word or
short phrase in each row:

``` r
sentences <- c("The apple doesn't fall far from the tree.", 
               "Don't judge a book by its cover.")
(df_sent <- strsplit(x = sentences, split = " ") |> 
  map_dfr(.f =  ~ data.frame(word = .x), .id = "sent_n"))
#> # A tidytable: 15 × 2
#>    sent_n word   
#>     <int> <chr>  
#>  1      1 The    
#>  2      1 apple  
#>  3      1 doesn't
#>  4      1 fall   
#>  5      1 far    
#>  6      1 from   
#>  7      1 the    
#>  8      1 tree.  
#>  9      2 Don't  
#> 10      2 judge  
#> 11      2 a      
#> 12      2 book   
#> 13      2 by     
#> 14      2 its    
#> 15      2 cover.
```

One can get the log-transformed probability of each word based on GPT-2
as follows:

``` r
df_sent <- df_sent |>
  mutate(lp = causal_words_pred(word, by = sent_n))
#> Processing using causal model 'gpt2/' ...
#> Processing a batch of size 1 with 10 tokens.
#> Processing a batch of size 1 with 9 tokens.
#> Text id: 1
#> `The apple doesn't fall far from the tree.`
#> Text id: 2
#> `Don't judge a book by its cover.`
#> ***
df_sent
#> # A tidytable: 15 × 3
#>    sent_n word         lp
#>     <int> <chr>     <dbl>
#>  1      1 The      NA    
#>  2      1 apple   -10.9  
#>  3      1 doesn't  -5.50 
#>  4      1 fall     -3.60 
#>  5      1 far      -2.91 
#>  6      1 from     -0.745
#>  7      1 the      -0.207
#>  8      1 tree.    -1.58 
#>  9      2 Don't    NA    
#> 10      2 judge    -6.27 
#> 11      2 a        -2.33 
#> 12      2 book     -1.97 
#> 13      2 by       -0.409
#> 14      2 its      -0.257
#> 15      2 cover.   -1.38
```

## How to cite

``` r
citation("pangoling")
Users are encouraged to not only cite pangoling, but also the python
package `transformers` (and the specific LLM they are using):

  Nicenboim B (2025-04-07 17:00:02 UTC). _pangoling: Access to large
  language model predictions in R_. doi:10.5281/zenodo.7637526
  <https://doi.org/10.5281/zenodo.7637526>, R package version 1.0.3,
  <https://github.com/ropensci/pangoling>.

  Wolf T, Debut L, Sanh V, Chaumond J, Delangue C, Moi A, Cistac P,
  Rault T, Louf R, Funtowicz M, Davison J, Shleifer S, von Platen P, Ma
  C, Jernite Y, Plu J, Xu C, Le Scao T, Gugger S, Drame M, Lhoest Q,
  Rush AM (2020). "HuggingFace's Transformers: State-of-the-art Natural
  Language Processing." 1910.03771, <https://arxiv.org/abs/1910.03771>.

To see these entries in BibTeX format, use 'print(<citation>,
bibtex=TRUE)', 'toBibtex(.)', or set
'options(citation.bibtex.max=999)'.
```

## How to contribute

See the [Contributing guidelines](.github/CONTRIBUTING.md).

## Code of conduct

Please note that this package is released with a [Contributor Code of
Conduct](https://ropensci.org/code-of-conduct/). By contributing to this
project, you agree to abide by its terms.

## See also

Another R package that act as a wrapper for
[`transformers`](https://pypi.org/project/transformers/) is
[`text`](https://r-text.org//) However, `text` is more general, and its
focus is on Natural Language Processing and Machine Learning.

[^1]: The logo of the package was created with [stable
    diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion)
    and the R package
    [hexSticker](https://github.com/GuangchuangYu/hexSticker).
