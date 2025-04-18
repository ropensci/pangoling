% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tr_utils.R
\name{tokenize_lst}
\alias{tokenize_lst}
\title{Tokenize an input}
\usage{
tokenize_lst(
  x,
  decode = FALSE,
  model = getOption("pangoling.causal.default"),
  add_special_tokens = NULL,
  config_tokenizer = NULL
)
}
\arguments{
\item{x}{Strings or token ids.}

\item{decode}{Logical. If \code{TRUE}, decodes the tokens into human-readable
strings, handling special characters and diacritics. Default is
\code{FALSE}.}

\item{model}{Name of a pre-trained model or folder. One should be able to use
models based on "gpt2". See
\href{https://huggingface.co/models?other=gpt2}{hugging face website}.}

\item{add_special_tokens}{Whether to include special tokens. It has the
same default as the
\href{https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoTokenizer}{AutoTokenizer}
method in Python.}

\item{config_tokenizer}{List with other arguments that control how the
tokenizer from Hugging Face is accessed.}
}
\value{
A list with tokens
}
\description{
Tokenize a string or token ids.
}
\examples{
\dontshow{if (installed_py_pangoling()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
tokenize_lst(x = c("The apple doesn't fall far from the tree."), 
             model = "gpt2")
\dontshow{\}) # examplesIf}
}
\seealso{
Other token-related functions: 
\code{\link{ntokens}()},
\code{\link{transformer_vocab}()}
}
\concept{token-related functions}
