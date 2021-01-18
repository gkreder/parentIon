################################################################################
# Gabe Reder - gkreder@gmail.com
################################################################################
# A python wrapper for running XCMS on input data (using XCMS 3 syntax)
################################################################################
import sys
import os
import pandas as pd
import numpy as np
import argparse
import hashlib
################################################################################
parser = argparse.ArgumentParser()

# Required file and parameter inputs 
parser.add_argument('--in_file', required = True)
args = parser.parse_args()
################################################################################

df = pd.read_excel(args.in_file, index_col = 0, names = ['name', 'value'], header = None)
out_dir = os.path.dirname(df.loc['out_tsv'].value)
os.system(f'mkdir -p {out_dir}')
# Create a random suffix for the temporary R file
hash_tag = str(int(hashlib.sha256(df.loc['out_tsv'].value.encode('utf-8')).hexdigest(), 16) % 10**12)

out_string = ""
files = ','.join([f'''"{x}"''' for x in df.loc['in_files'].value.split(',')])
sample_names = ','.join([f'''"{x}"''' for x in df.loc['sample_names'].value.split(',')])
if str(df.loc['sample_groups']) != "None":
    sample_groups = ','.join([f'''"{x}"''' for x in df.loc['sample_groups'].value.split(',')])
else:
    sample_groups = "integer(length(sample_names))"
out_string += f'''library(xcms)
library(plyr)
library(dplyr)
library(stringr)
options(dplyr.summarise.inform = FALSE)
BiocParallel::register(BiocParallel::SerialParam())
files <- c({files})
sample_names <- c({sample_names})
sample_groups <- c({sample_groups})
pd <- data.frame(sample_name = sample_names, sample_group = sample_groups, stringsAsFactors = FALSE)
raw_data <- readMSData(files = files, pdata = new("NAnnotatedDataFrame", pd), mode = "onDisk")
cwp <- CentWaveParam(ppm = {df.loc['centwave_ppm'].value}, mzdiff = {df.loc['centwave_mzdiff'].value}, integrate = {df.loc['centwave_integrate'].value}, fitgauss = {df.loc['centwave_fitgauss'].value}, noise = {df.loc['centwave_noise'].value}, peakwidth=c({df.loc['centwave_peakwidth'].value}), prefilter = c({df.loc['centwave_prefilter'].value}), snthresh = {df.loc['centwave_snthresh'].value}, mzCenterFun = "{df.loc['centwave_mzCenterFun'].value}")
xdata <- findChromPeaks(raw_data, param = cwp)
owp <- ObiwarpParam(factorGap = {df.loc['obiwarp_factorGap'].value} , binSize = {df.loc['obiwarp_binSize'].value}, factorDiag = {df.loc['obiwarp_factorDiag'].value} , distFun = "{df.loc['obiwarp_distFun'].value}", response = {df.loc['obiwarp_response'].value} , localAlignment = FALSE, initPenalty = {df.loc['obiwarp_initPenalty'].value} )
pdp <- PeakDensityParam(sampleGroups = xdata$sample_group, minSamples = {df.loc['density_minSamples'].value}, minFraction = {df.loc['density_minFraction'].value}, binSize = {df.loc['density_binSize'].value}, bw = {df.loc['density_bw'].value}, maxFeatures = {df.loc['density_maxFeatures'].value})
xdata <- adjustRtime(xdata, param=owp)
xdata <- groupChromPeaks(xdata, param = pdp)
xdata <- adjustRtime(xdata, param=owp)
xdata <- groupChromPeaks(xdata, param = pdp)
xdata <- adjustRtime(xdata, param=owp)
xdata <- groupChromPeaks(xdata, param = pdp)
xdata <- fillChromPeaks(xdata)
fd <- featureDefinitions(xdata)
fv <- featureValues(xdata, value = 'into')
# fs <- featureSummary(xdata, group = xdata$sample_group)
pd <- phenoData(xdata)
cp <- chromPeaks(xdata)
sample_names <- pd@data$sample_name
peakidxs <- fd$peakidx'''
out_string +='''
suppressWarnings(feature_mzs <- lapply(peakidxs, function (x) {
	if (length(x) == 1){
		cp_x <- cp[x, ]
		sn <- sample_names[cp_x['sample']]
		concat_mz <- data.frame(temp_name = cp_x['mz'])
		names(concat_mz) <- paste('mz', sn, sep = '_')
	}
	else {
		cp_x <- data.frame(cp[x, ])
		sns <- sample_names[cp_x$sample]
		cp_x$sample_name <- sns

		concat_mz <- cp_x %>% dplyr::group_by(sample_name) %>% dplyr::summarise(mz = toString(mz)) # , rt = toString(rt)
		cn_mz <- lapply(concat_mz$sample_name, function (x) { paste('mz', x, sep = '_') })
		concat_mz <- data.frame(t(concat_mz[, -1]))
		colnames(concat_mz) <- cn_mz
	}
	concat_mz
}) %>% plyr::rbind.fill())
feature_mzs$feature_id <- row.names(fv)
feature_mzs <- feature_mzs[, c("feature_id", paste("mz_", sample_names, sep = ""))]

suppressWarnings(feature_rts <- lapply(peakidxs, function (x) {
	if (length(x) == 1){
		cp_x <- cp[x, ]
		sn <- sample_names[cp_x['sample']]
		concat_rt <- data.frame(temp_name = cp_x['rt'])
		names(concat_rt) <- paste('rt', sn, sep = '_')
	}
	else {
		cp_x <- data.frame(cp[x, ])
		sns <- sample_names[cp_x$sample]
		cp_x$sample_name <- sns

		concat_rt <- cp_x %>% dplyr::group_by(sample_name) %>% dplyr::summarise(rt = toString(rt))
		cn_rt <- lapply(concat_rt$sample_name, function (x) { paste('rt', x, sep = '_') })
		concat_rt <- data.frame(t(concat_rt[, -1]))
		colnames(concat_rt) <- cn_rt
	}
	concat_rt
}) %>% plyr::rbind.fill())
feature_rts$feature_id <- row.names(fv)
feature_rts <- feature_rts[, c("feature_id", paste("rt_", sample_names, sep = ""))]

merged_data <- data.frame(merge(fd, fv, by = 'row.names'))
colnames(merged_data)[1] <- "feature_id"
merged_data <- merge(merged_data, feature_rts, by = 'feature_id')
merged_data <- merge(merged_data, feature_mzs, by = 'feature_id')
merged_data <- merged_data %>% dplyr::mutate(name = paste("M", round(mzmed), 'T', round(rtmed), sep="")) %>% select(name, everything())

merged_data$peakidx <- str_replace_all(merged_data$peakidx, "[\\r\\n]" , "")

names <- as.character(merged_data$name)
merged_data$name <- ifelse(duplicated(names) | duplicated(names, fromLast=TRUE), paste(names, ave(names, names, FUN=seq_along), sep='_'), names)'''

out_string += f'''
write.table(merged_data, '{df.loc['out_tsv'].value}', sep='\\t', col.names=NA)
cpData <- chromPeaks(xdata)
cpData <- write.table(cpData, '{df.loc['out_tsv'].value.replace('.tsv', '_CPDATA.tsv')}', sep='\\t', col.names=NA)
saveRDS(xdata, file = '{df.loc['out_tsv'].value.replace('.tsv', '_XCMSSET.rds')}')
'''

with open(f"xcms_{hash_tag}.R", 'w') as f:
    print(out_string, file = f)
os.system(f"Rscript xcms_{hash_tag}.R")
os.system(f"rm xcms_{hash_tag}.R")

