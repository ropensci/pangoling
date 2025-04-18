% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tr_causal.R
\name{causal_next_tokens_pred_tbl}
\alias{causal_next_tokens_pred_tbl}
\title{Generate next tokens after a context and their predictability using a causal
transformer model}
\usage{
causal_next_tokens_pred_tbl(
  context,
  log.p = getOption("pangoling.log.p"),
  decode = FALSE,
  model = getOption("pangoling.causal.default"),
  checkpoint = NULL,
  add_special_tokens = NULL,
  config_model = NULL,
  config_tokenizer = NULL
)
}
\arguments{
\item{context}{A single string representing the context for which the next
tokens and their predictabilities are predicted.}

\item{log.p}{Base of the logarithm used for the output predictability values.
If \code{TRUE} (default), the natural logarithm (base \emph{e}) is used.
If \code{FALSE}, the raw probabilities are returned.
Alternatively, \code{log.p} can be set to a numeric value specifying
the base of the logarithm (e.g., \code{2} for base-2 logarithms).
To get surprisal in bits (rather than predictability), set
\code{log.p = 1/2}.}

\item{decode}{Logical. If \code{TRUE}, decodes the tokens into human-readable
strings, handling special characters and diacritics. Default is
\code{FALSE}.}

\item{model}{Name of a pre-trained model or folder. One should be able to use
models based on "gpt2". See
\href{https://huggingface.co/models?other=gpt2}{hugging face website}.}

\item{checkpoint}{Folder of a checkpoint.}

\item{add_special_tokens}{Whether to include special tokens. It has the
same default as the
\href{https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoTokenizer}{AutoTokenizer}
method in Python.}

\item{config_model}{List with other arguments that control how the
model from Hugging Face is accessed.}

\item{config_tokenizer}{List with other arguments that control how the
tokenizer from Hugging Face is accessed.}
}
\value{
A table with possible next tokens and their log-probabilities.
}
\description{
This function predicts the possible next tokens and their predictability
(log-probabilities by default). The function sorts tokens in descending order
of their predictability.
}
\details{
The function uses a causal transformer model to compute the predictability
of all tokens in the model's vocabulary, given a single input context. It
returns a table where each row represents a token, along with its
predictability score. By default, the function returns log-probabilities in
natural logarithm (base \emph{e}), but you can specify a different logarithm base
(e.g., \code{log.p = 1/2} for surprisal in bits).

If \code{decode = TRUE}, the tokens are converted into human-readable strings,
handling special characters like accents and diacritics. This ensures that
tokens are more interpretable, especially for languages with complex
tokenization.
}
\section{More details about causal models}{

A causal language model (also called GPT-like, auto-regressive, or decoder
model) is a type of large language model usually used for text-generation
that can predict the next word (or more accurately in fact token) based
on a preceding context.

If not specified, the causal model used will be the one set in the global
option \code{pangoling.causal.default}, this can be
accessed via \code{getOption("pangoling.causal.default")} (by default
"gpt2"). To change the default option
use \code{options(pangoling.causal.default = "newcausalmodel")}.

A list of possible causal models can be found in
\href{https://huggingface.co/models?pipeline_tag=text-generation}{Hugging Face website}.

Using the  \code{config_model} and \code{config_tokenizer} arguments, it's possible to
control how the model and tokenizer from Hugging Face is accessed, see the
Python method
\href{https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained}{\code{from_pretrained}}
for details.

In case of errors when a new model is run, check the status of
\url{https://status.huggingface.co/}
}

\examples{
\dontshow{if (installed_py_pangoling()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
causal_next_tokens_pred_tbl(
  context = "The apple doesn't fall far from the",
  model = "gpt2"
)
\dontshow{\}) # examplesIf}
}
\seealso{
Other causal model functions: 
\code{\link{causal_pred_mats}()},
\code{\link{causal_words_pred}()}
}
\concept{causal model functions}
