## Resubmission

This is a resubmission. 

> Please write references in the description of the DESCRIPTION file in the form
> authors (year) <doi:...>
> authors (year, ISBN:...)
> or if those are not available: authors (year) <https:...>
> with no space after 'doi:', 'https:' and angle brackets for auto-linking. (If you want to add a title as well please put it in quotes: "Title") -> please add some form of linking to Radford et al., 2019 and write the years in parentheses.
> For more details: <https://contributor.r-project.org/cran-cookbook/description_issues.html#references>

**Answer**:
This has been fixed.

> Please provide a link to the used webservices to the description field of your DESCRIPTION file in the form
> <http:...> or <https:...>
> with angle brackets for auto-linking and no space after 'http:' and 'https:'.
> For more details: <https://contributor.r-project.org/cran-cookbook/description_issues.html#references>

**Answer**:
I have added the link <https://huggingface.co/> to the description

> You have examples wrapped in if(FALSE). Please never do that. Ideally find toy examples that can be regularly executed and checked. Lengthy examples (> 5 sec), can be wrapped in \donttest{}. \dontrun{} can be used if the example really cannot be executed (e.g. because of missing additional software, missing API keys, ...) by the user.

**Answer**:
There was one example that trigger the installation of python packages wrapped in `if(FALSE)`. This has been changed to `\dontrun{}`. But notice that very few examples can be actually run, since most functions depend on additional software (python packages). 


> You write information messages to the console that cannot be easily suppressed.
> It is more R like to generate objects that can be used to extract the information a user is interested in, and then print() that object. Instead of cat() rather use message()/warning() or if(verbose)cat(..) (or maybe stop()) if you really have to write text to the console. (except for print, summary, interactive functions) -> R/utils.R
For more details: <https://contributor.r-project.org/cran-cookbook/code_issues.html#using-printcat>

**Answer**:
I changed the `cat()` call to `message()` through the wrapper function `message_verbose()`.

> Please do not modify the global environment (e.g. by using <<-) in your functions. This is not allowed by the CRAN policies. -> R/zzz.R

**Answer**:
`.Onload` function  had:
`inspect <<- reticulate::import("inspect", delay_load = TRUE, convert = TRUE)`
but no corresponding `inspect <- NULL` call in the parent scope, which caused `<<-` to continue searching parent environments until it reaches the `globalenv.`
I removed it. (I also removed `transformers <<-` because it wasn't necessary.)

Notice that `.Onload` function in zzz.R still has 
`torch <<- reticulate::import("torch", delay_load = TRUE, convert = FALSE)`
but there is a corresponding `torch <- NULL` call in the parent scope.

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.



