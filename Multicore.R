options(warn=-1)
library(snow)
library(rlecuyer)

# Set output directory
args <- commandArgs(TRUE)

# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (image dir)", call.=FALSE)
} else if (length(args)==1) {
  # default output file
  args[2] = "out.txt"
}

#print(args[1])
#print(args[2])
#print(args[3])
#print(args[4])
#input1 <- toString(args[1])
#output1 <- toString(args[2])
#pythonscript1 <- toString(args[3])
#batchdir1 <- toString(args[4])

#output <- 'D:/pinapple/smalltest/Output'
#obia <- 'D:/pinapple/SegmentAndClass/OBIA_chain_v2.py'
#segment <- 'D:/pinapple/SegmentAndClass/LSMSSegmentation_chain.bat'

# Files to process
filenames <- invisible(list.files(args[1], pattern=".tif$", full.names=T))

#startpython3 <- paste("python", args[3], '--output', args[2], '--bat', args[4], '--pre yes --att rgbnir')
#startpython2 <- paste("python", obia, '--output', output, '--bat', segment, '--pre yes --att rgbnir')
#print(startpython2)


# Process function
processfunc <- function(filename, filenames, out = 'D:/pinapple/smalltest/Output', pythonscript = 'D:/pinapple/SegmentAndClassPin/Main.py', batch = 'D:/pinapple/SegmentAndClassPin/LSMSSegmentation_chain.bat'){
  len_names <- length(filenames) #Length of filenames
  index <- match(filename, filenames) #Index of current file
  system(paste("echo", "'Processing tiles:",format(round(index/len_names*100,2),nsmall=2),"% At:",filename,"'")) # print progress
  system(paste('python', pythonscript, '--output', out, '--bat', batch, '--pre yes --att rgbnir --image', filenames))
}

corenr=4
cl = invisible(makeCluster(rep('localhost', corenr), 'SOCK'))
invisible(clusterExport(cl, list("processfunc")))
invisible(clusterEvalQ(cl, library(stringr))) 
invisible(clusterSetupRNG(cl))
invisible(clusterMap(cl,function(x, filenames) processfunc(x, filenames), filenames, MoreArgs = list(filenames = filenames)))
invisible(stopCluster(cl))
