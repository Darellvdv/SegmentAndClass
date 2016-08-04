options(warn=-1)
library(snow)
library(rlecuyer)

# Set output directory
args <- commandArgs(TRUE)


# Test if all the variables are used
if (length(args)==10) {
  stop("At least one of the essential settings is missing", call.=FALSE)
} else if (length(args)==1) {
  # default output file
  args[2] = "D:/tempout"
}

# Files to process
filenames <- invisible(list.files(args[1], pattern=".tif$", full.names=T))


if (length(args)==11) {
  
  # Process function
  processfunc <- function(filename, filenames, pythonscript, out, batch, att, pre, rang, spatialr, minsiz, tilesize){
    system(paste('python', pythonscript, '--output', out, '--bat', batch, '--pre', pre, '--att', att, '--image', filename, 
                 '--rang', rang, '--spatialr', spatialr, '--minsiz', minsiz, '--tilesize', tilesize))
    }
  corenr <<- as.integer(args[11])
  cl = invisible(makeCluster(rep('localhost', corenr), 'SOCK'))
  invisible(clusterExport(cl, list("processfunc")))
  invisible(clusterEvalQ(cl, library(stringr))) 
  invisible(clusterSetupRNG(cl))
  invisible(clusterMap(cl,function(x, filenames, pythonscript, out, batch, att, pre, rang, spatialr, minsiz, tilesize) 
  
  processfunc(x, filenames, pythonscript, out, batch, att, pre, rang, spatialr, minsiz, tilesize), 
  filenames, MoreArgs = list(filenames = filenames,pythonscript = args[3],out = args[2],batch = args[4],att = args[5],
  pre = args[6],rang = args[7],spatialr = args[8],minsiz = args[9],tilesize = args[10])))
  
  invisible(stopCluster(cl))

} else {
  
  # Process function
  processfunc <- function(filename, filenames, pythonscript, out, batch, att, pre, rang, spatialr, minsiz, tilesize, dsm, dtm){
    system(paste('python', pythonscript, '--output', out, '--bat', batch, '--pre', pre, '--att', att, '--image', filename, 
                 '--rang', rang, '--spatialr', spatialr, '--minsiz', minsiz, '--tilesize', tilesize, '--dsm', dsm, '--dtm', dtm))
  }
  corenr <<- as.integer(args[11])
  cl = invisible(makeCluster(rep('localhost', corenr), 'SOCK'))
  invisible(clusterExport(cl, list("processfunc")))
  invisible(clusterEvalQ(cl, library(stringr))) 
  invisible(clusterSetupRNG(cl))
  invisible(clusterMap(cl,function(x, filenames, pythonscript, out, batch, att, pre, rang, spatialr, minsiz, tilesize, dsm, dtm) 
    processfunc(x, filenames, pythonscript, out, batch, att, pre, rang, spatialr, minsiz, tilesize, dsm, dtm), 
    filenames, MoreArgs = list(filenames = filenames,
                               pythonscript = args[3],
                               out = args[2],
                               batch = args[4],
                               att = args[5],
                               pre = args[6],
                               rang = args[7],
                               spatialr = args[8],
                               minsiz = args[9],
                               tilesize = args[10],
                               dsm = args[12],
                               dtm = args[13])))
  
  invisible(stopCluster(cl))
}
