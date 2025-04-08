# Pre-compiled vignettes 
knitr::knit("vignettes/example.Rmd.orig", "vignettes/example.Rmd")
knitr::knit("vignettes/intro-bert.Rmd.orig", "vignettes/intro-bert.Rmd")
knitr::knit("vignettes/intro-gpt2.Rmd.orig", "vignettes/intro-gpt2.Rmd")
destination_folder <- "vignettes"
files <- list.files(pattern = "vigfig-")
file.rename(files, file.path(destination_folder, basename(files)))
# change bibliography manually!!!!

