################################################################################
# Gabe Reder - gkreder@gmail.com
################################################################################
import pickle as pkl
import sys
import os
import argparse
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
import numpy as np
import ast
import itertools
import sklearn.metrics
import scipy.spatial
import scipy.cluster
import warnings
import matplotlib.pyplot as plt

import rpy2.robjects as ro
import rpy2.robjects.packages as packages
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()

rbase = packages.importr('base')
rstats = packages.importr('stats')
qvalue = packages.importr('qvalue')
################################################################################
# Collecting arguments and initializing logfile / output directory
################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--in_file', required = True)
args = parser.parse_args()

if args.in_file.endswith('.xlsx'):
	df_args = pd.read_excel(args.in_file, names = ['arg', 'value'], header = None)
elif args.in_file.endswith('.tsv'):
	df_args = pd.read_csv(args.in_file, sep = '\t', names = ['arg', 'value'], header = None)
else:
	sys.exit(f"Error - unrecognized input file extension {args.in_file.split('.')[-1]} try one of [.xlsx, .tsv]")
	
for arg, value in df_args.values:
	if str(value) == 'nan':
		value = None
	setattr(args, arg, value)


# Make sure sample_groups, sample_names, and rt_cols are provided as comma-delimited
# lists
if args.sample_groups != None:
	args.sample_groups = [x.replace(' ', '') for x in args.sample_groups.split(",")]
args.sample_names = [x.replace(' ', '') for x in args.sample_names.split(",")]
args.rt_cols = [x.replace(' ', '') for x in args.rt_cols.split(",")]
args.no_recursive_clustering = {'false' : False, 'true' : True}[str(args.no_recursive_clustering).lower()]


# Set up output directory, logfile, and intermediate files directory
args.out_prefix = os.path.basename(args.out_tsv).replace('.tsv', '')
args.out_dir = os.path.dirname(args.out_tsv)
args.intermediate_files_dir = os.path.join(args.out_dir, args.out_prefix + '_clustering_files')
os.system('mkdir -p %s' % args.intermediate_files_dir)

logfile_name = args.out_tsv.replace('.tsv', '.log')
logfile = open(logfile_name, 'w')
print('python ' + ' '.join(sys.argv), file = logfile)
print(args, file = logfile)
print('', file = logfile)

################################################################################
# Read input files and process
################################################################################
time_stamp = datetime.now()
print('########################################################################')
print('Reading input files and processing... %s' % time_stamp)
print('########################################################################')
sys.stdout.flush()

df = pd.read_csv(args.in_tsv, sep = '\t').set_index('name')
# The behavior below calculates intensity ranks, ignoring NaNs (no feature value)
# It does not fill the NaNs in as 0.0, to do this uncomment the line below
intensity_ranks = df[args.sample_names].mean(axis = 1, skipna = True).sort_values(ascending = False).index
# intensity_ranks = df[args.sample_names].fillna(value = 0.0).mean(axis = 1, skipna = True).sort_values(ascending = False).index
df_sorted_intensities = df[args.sample_names].loc[intensity_ranks]
df_sorted_rts = df[args.rt_cols].loc[intensity_ranks]

# Auxiliary function for interpreting the intensity and retention time values that are 
# reported as a list of values rather than a single one (product of XCMS output)
def merge_func(x):
	try:
		m = np.mean(ast.literal_eval(x))
	except:
		m = x
	return(m)

df_sorted_intensities = np.log10(df_sorted_intensities.applymap(merge_func).replace(to_replace = 0, value = 1.0, inplace = False))
df_sorted_rts = df_sorted_rts.applymap(merge_func)

print('done...%s' % str(datetime.now() - time_stamp))
################################################################################
# Calculate Linkage
################################################################################
# The linkage is calculated based on a custom distance metric. For two XCMS
# features, fi and fj, we calculate the distance metric as:
#					d(fi, fj) = (1 - Rij) + ( alpha * (1 - exp(-pij / tau)) ) 
# where Rij is the pearson correlation distance between fi and fj intensities
# across samples (in overlapping samples only). And pij looks at the retention
# time distance between fi and fj pairwise calculated across overlapping samples:
#					pij = sqrt( (1 / n) * sum(tik - tjk)^2)
# where tik is fi's retention time for sample k. 
################################################################################
time_stamp = datetime.now()
print('########################################################################')
print('Calculating linkage... %s' % time_stamp)
print('########################################################################')
sys.stdout.flush()

# Calculation of the pairwise distance matrix for our custom 
# metric. Currently optimized for runtime, not memory usage. Requires large
# amounts of RAM for feature sets that are too large
def get_dist_mat(df_sorted_intensities, df_sorted_rts, max_distance):
	if df_sorted_intensities.shape != df_sorted_rts.shape:
		sys.exit(f"Error in dist_mat - df_sorted_intensities shape {df_sorted_intensities.shape} must be equal to df_sorted_rts shape {df_sorted_rts.shape}")
	num_cols = df_sorted_rts.shape[1]
	# Row-vs-row calculations of where intensity/rt values for a given feature exist in both 
	# samples (df_noNans) or none of the samples (df_bothNans)
	df_noNans = (~df_sorted_intensities.isna()).astype(int).dot((~df_sorted_intensities.isna()).astype(int).T)
	df_bothNans = (df_sorted_intensities.isna()).astype(int).dot((df_sorted_intensities.isna()).astype(int).T)
	
	# Calculates feature-vs-feature Pearson distance, ignoring entries (samples) with NaN intensity
	# in either of the two given features for a single calculation
	pDists = (1 - df_sorted_intensities.T.corr(method = 'pearson')).values
	# calculate manhattan distances for edge cases where there are only one or two non-nan overlaps 
	# and everything else is a nan-overlap
	mDists = sklearn.metrics.pairwise.manhattan_distances(df_sorted_intensities.fillna(value = 0.0))

	# For distance(feature1, feature2), if 0 < overlap_samples(feature1, feature2) < 3 and 
	# overlap_missingSamples(feature1, feature2) >= nonOverlapping_samples(feature1, feature2)...
	# 		1st case - the few intensity observations are the same for feature1 and feature2, they are the "same"
	pDists[np.where((df_noNans > 0) & (df_noNans < 3) & ((num_cols - df_noNans) <= df_bothNans) & (mDists < 1e-2))] = 0.0
	# 		2nd case - the few intensity observations aren't exactly the same, set them to be very close to each other
	# 		at 0.1 since they share a couple of sparse observations and concordant missing observations
	pDists[np.where((df_noNans > 0) & (df_noNans < 3) & ((num_cols - df_noNans) <= df_bothNans) & (mDists >= 1e-2))] = 0.1

	# Remaining nan values in pDists should be coming from situations in which either one or both rows of observations
	# consist of multiple observations of the exact same value (likely 0.0), making the denominator in the Pearson
	# Coefficient 0. If Manhattan distance is close enough, set pDist = 0 if not set to max_distance / 2
	pDists[np.where(np.isnan(pDists) & (mDists < 1e-2))] = 0.0
	pDists[np.where(np.isnan(pDists) & (mDists >= 1e-2))] = max_distance / 2.0
	
	# Calculate the retention-time portio of the metric. Again, ignoring NaN cells in the calculation
	rDists = np.round(np.sqrt( sklearn.metrics.pairwise.nan_euclidean_distances(df_sorted_rts.values, squared = True) / num_cols), 3)
	if args.tau == 0.0:
		rDists = (np.ones(rDists.shape) - (np.nan_to_num(np.abs(rDists), nan = np.inf) <= 1e-4).astype(int)) * args.alpha
	else:
		rDists = args.alpha * (1 - np.exp(-rDists / args.tau))
	
	allDists = pDists + rDists
	# If there are 0 overlapping samples for feature1 and feature2, set them
	# to max distance from each other
	allDists[np.where(df_noNans == 0)] = max_distance
	# If there are 1 or 2 overlapping samples for feature1 and feature2  and 
	# overlap_missingSamples(feature1, feature2) < nonOverlapping_samples(feature1, feature2), set them
	# to max distance from each other
	allDists[np.where((df_noNans > 0) & (df_noNans < 3) & ((num_cols - df_noNans) > df_bothNans))] = max_distance

	return(pd.DataFrame(allDists, index = df_sorted_intensities.index, columns = df_sorted_intensities.index))

def get_linkage(df_sorted_intensities, df_sorted_rts, max_distance):
	dist_mat = get_dist_mat(df_sorted_intensities, df_sorted_rts, max_distance)
	dist_diffs = dist_mat - dist_mat.T
	max_diff = dist_diffs.to_numpy().max()
	# Manually check that the distance matrix is symmetric and has 
	# zeros down the diagonal because the checks are turned off in 
	# squareform call below
	if max_diff > 1e-4:
		max_row = dist_diffs.max().sort_values(ascending = False).index[0]
		max_col = dist_diffs[max_row].sort_values(ascending = False).index[0]    
		sys.exit(f'Error - alpha={args.alpha} tau={args.tau}, the distance calculation did not produce a symmetric matrix. E.g. check the calculation at {max_row}, {max_col} with a calculation difference of {dist_diffs[max_row][max_col]}')
	if np.any(np.diag(dist_mat)):
		sys.exit(f'Error - alpha={args.alpha} tau={args.tau}, the distance calculation did not produce zeros on the diagonal of the distance matrix. E.g. check {dist_mat.index[np.where(np.diag(dist_mat))[0][0]]}')
	dist_mat_1D = scipy.spatial.distance.squareform(dist_mat, checks = False)
	return(scipy.cluster.hierarchy.linkage(dist_mat_1D, method = 'average'))


linkage = get_linkage(df_sorted_intensities, df_sorted_rts, max_distance = 2 + args.alpha)

print('done...%s' % str(datetime.now() - time_stamp))
################################################################################
# Find clusters
################################################################################
# Now that we have a linkage tree (hierarchical clustering dendrogram), can 
# compute the tree cuts to find the initial clusters. Clusters are defined as 
# tree cuts where args.frac_peaks of the resulting leaves are within 
# args.rt_1sWindow of the root RT (defined by rt_distance())
################################################################################
time_stamp = datetime.now()
print('########################################################################')
print('Finding clusters... %s' % time_stamp)
print('########################################################################')
sys.stdout.flush()


tree = scipy.cluster.hierarchy.to_tree(linkage)

# pre-compute the rts and save them to leaf objects in tree
def set_rt(leaf, df_sorted_rts):
	# rt = df_sorted_rts.iloc[leaf.id].values[0]
	rt = df_sorted_rts.iloc[leaf.id]
	leaf.rt = rt
_ = tree.pre_order(lambda leaf : set_rt(leaf, df_sorted_rts))

# indices already correspond to intensity sort order since linkage was calculated
# on the df_sorted_intensities values
leaf_list = sorted(tree.pre_order(lambda leaf : (leaf.id, leaf.rt)), key = lambda tup : tup[0])

# The retention time distance between two features is found using the median absolute 
# difference between RTs for features across all samples. If two features do not 
# have any overlapping samples, rt_distance is defined using args.rt_1sWindow
def rt_distance(rts1, rts2):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)
		r = np.nan_to_num(np.nanmedian(np.abs(rts1 - rts2)), nan = args.rt_1sWindow + 1.0)
	return(r)

# Recursive function for getting all the cuts
def get_cuts(tree, rt_1sWindow, frac_peaks):
	leaf_list = sorted(tree.pre_order(lambda leaf : (leaf.id, leaf.rt)), key = lambda tup : tup[0])
	current_rt = leaf_list[0][1]
	good_peaks = np.sum([rt_distance(current_rt, x[1]) <= rt_1sWindow for x in leaf_list]) / len(leaf_list)
	
	# If we are far enough down the tree where the reuslting sub-tree contains "good peaks" above the 
	# desired fraction (frac_peaks), then stop here and return the current sub-tree
	if good_peaks >= frac_peaks:
		parent_id = leaf_list[0][0]
		cluster_ids = [x[0] for x in leaf_list]
		return([(parent_id, cluster_ids)])
	# If we're not far enough down the tree to hit the stopping criterion, keep going down the tree and 
	# return the sub-trees found for the left- and right- calls
	else:
		return(get_cuts(tree.get_right(), rt_1sWindow, frac_peaks) + get_cuts(tree.get_left(), rt_1sWindow, frac_peaks))

# Starting at the top, find all the cuts that satisfy the stopping criterion (trees where args.frac_peaks are within 
# args.rt_1sWindow of the root defined using rt_distance)
cut_ids = get_cuts(tree, rt_1sWindow = args.rt_1sWindow, frac_peaks = args.frac_peaks)
clusters = [(df_sorted_intensities.index[p_id], p_id, clust_ids, [df_sorted_intensities.index[i] for i in clust_ids]) for p_id, clust_ids in cut_ids]

df_cuts = pd.DataFrame(clusters, columns=['peak_name', 'sorted_index', 'cluster_subtree_ids', 'cluster_subtree_names'])
df_cuts.to_csv(os.path.join(args.intermediate_files_dir, args.out_prefix + '_CUTS.tsv'), sep = '\t', index = False)

print('done...%s' % str(datetime.now() - time_stamp))
################################################################################
# Merge Clusters
################################################################################
# Using the cuts found above, translate them into clusters, saivng feature mz 
# and rt information
################################################################################
time_stamp = datetime.now()
print('########################################################################')
print('Merging clusters... %s' % time_stamp)
print('########################################################################')
sys.stdout.flush()


clusters = []
for i_p, peak_name in enumerate(tqdm(df_cuts['peak_name'])):
	p_clust = df_cuts['cluster_subtree_names'].values[i_p]
	
	clust_size = len(p_clust)
	clust_intensities = []
	clust_rts = []
	clust_mzs = []
	for p in p_clust:
		clust_mzs.append(df.loc[p]['mzmed'])
		clust_intensities.append(df_sorted_intensities.loc[p].to_dict())
		# clust_rts.append(df_sorted_rts.loc[p]['rtmed'])
		clust_rts.append(df.loc[p]['rtmed'])
	
	clusters.append((peak_name, p_clust, clust_intensities, clust_mzs, clust_rts, clust_size))

df_merged = pd.DataFrame(clusters, columns=['parent_peak', 'cluster', 'clust_intensities', 'clust_mzs', 'clust_rts', 'clust_size'])
df_merged.to_csv(os.path.join(args.intermediate_files_dir, args.out_prefix + '_MERGED.tsv'), sep = '\t', index = False)


print('done...%s' % str(datetime.now() - time_stamp))
################################################################################
# Finalize Clusters
################################################################################
# For every cluster, kick out extreme RT outliers 
# (defined by args.cluster_outlier_1swidth). Also, if there exists some leaf
# with an m/z greater than the cluster's parent and at least 
# args.parent_mz_check_intensity_frac percent of the parent's intensity, 
# make that the new cluster parent
################################################################################
time_stamp = datetime.now()
print('########################################################################')
print('Finalizing clusters... %s' % time_stamp)
print('########################################################################')
sys.stdout.flush()

clusters = []

for i_p, parent_peak in enumerate(tqdm(df_merged['parent_peak'])):
	
	# RT Filtering (only keep peaks that fall within reasonable RT range)
	clust = df_merged['cluster'].values[i_p]
	clust_rts = df_merged['clust_rts'].values[i_p]

	parent_index = df_merged['cluster'].values[i_p].index(parent_peak)
	parent_rt = clust_rts[parent_index]
	lb = parent_rt - args.cluster_outlier_1sWidth
	ub = parent_rt + args.cluster_outlier_1sWidth

	keep_indices = [(r >= lb and r <= ub) for r in clust_rts] 

	dropped = [(i, c) for i, c in enumerate(clust) if not keep_indices[i]]

	# Kept Cluster
	clust = [c for i, c in enumerate(clust) if keep_indices[i]]
	clust_intensities = [d for i, d in enumerate(df_merged['clust_intensities'].values[i_p]) if keep_indices[i]]
	clust_mzs = [d for i, d in enumerate(df_merged['clust_mzs'].values[i_p]) if keep_indices[i]]
	clust_rts = [d for i, d in enumerate(clust_rts) if keep_indices[i]]
	clust_size = len(clust)

	clust_avg_intensities = [np.mean(df[args.sample_names].fillna(value = 0.0).loc[x]) for x in clust]
	parent_index = np.argmax(clust_avg_intensities)
	parent_peak = clust[parent_index]
	parent_mz = clust_mzs[parent_index]

	# Make sure we can get through the entire cluster without running into
	# a cluster child that should be the new parent (higher m/z than parent
	# and high enough intensity)
	restart = True
	while restart:
		restart = False
		for i_cm, cm in enumerate(clust_mzs):
			if clust_mzs[i_cm] > parent_mz:
				if clust_avg_intensities[i_cm] >= clust_avg_intensities[parent_index] * args.parent_mz_check_intensity_frac:
					parent_index = i_cm
					parent_peak = clust[i_cm]
					parent_mz = clust_mzs[i_cm]
					restart = True
					break
	
	
	
	clusters.append((parent_peak, clust, clust_intensities, clust_mzs, clust_rts, clust_size))

	

	############################################################################
	# Current implementation = every dropped element automatically becomes 
	# its own cluster
	############################################################################
	for i_dpn, dpn in dropped:
		dropped_clust = [dpn]
		dpn_intensities = [df_merged['clust_intensities'].values[i_p][i_dpn]]
		dpn_mzs = [df_merged['clust_mzs'].values[i_p][i_dpn]]
		dpn_rts = [df_merged['clust_rts'].values[i_p][i_dpn]]
		dpn_size = len(dropped_clust)


		clusters.append((dpn, dropped_clust, dpn_intensities, dpn_mzs, dpn_rts, dpn_size))
	
	
	
df_clusters = pd.DataFrame(clusters, columns=['parent_peak', 'cluster', 'clust_intensities', 'clust_mzs', 'clust_rts', 'clust_size'])


print('done...%s' % str(datetime.now() - time_stamp))
################################################################################
# Recursive Clustering
################################################################################
# Iteratively re-compute clusters using the same scheme as described above 
# using only the current parents. If two parents cluster together, merge
# their respective clusters. Continue until the resulting number of clusters
# remains stable
################################################################################
time_stamp = datetime.now()
print('########################################################################')
print('Recursive clustering... %s' % time_stamp)
print('########################################################################')
sys.stdout.flush()


pn_index = {v : i for i,v in enumerate(df_sorted_intensities.index)}
df_clusters.to_csv(os.path.join(args.intermediate_files_dir, args.out_prefix + '_RECURSIVE_ROUND_0.tsv'), sep = '\t', index = False)
linkage_backup = linkage
pkl.dump(linkage_backup, open(os.path.join(args.intermediate_files_dir, args.out_prefix + '_linkageBackup.pkl'), 'wb'))
leaves = scipy.cluster.hierarchy.leaves_list(linkage_backup)
dend_order = [df_sorted_intensities.index[x] for x in leaves]
num_clusters = len(df_clusters)
parent_peaks = df_clusters['parent_peak'].values
df_clusters = pd.DataFrame(df_clusters.set_index('parent_peak', drop = False))

######################################
# Recursive Clustering
######################################
# This contains some duplicate code
# from above during the recursive rounds.
# Should refactor everything into
# function calls
######################################

recursive_round_num = 1
while not args.no_recursive_clustering:

	print('Entering recursive clustering round %i - %i clusters (%s)' % (recursive_round_num, num_clusters, str(datetime.now())))
	print('Entering recursive clustering round %i - %i clusters (%s)' % (recursive_round_num, num_clusters, str(datetime.now())), file = logfile)
	sys.stdout.flush()
	
	df_sorted_intensities_rec = pd.DataFrame(df_sorted_intensities[df_sorted_intensities.index.isin(parent_peaks)])
	df_sorted_rts_rec = pd.DataFrame(df_sorted_rts[df_sorted_rts.index.isin(parent_peaks)])
	
	linkage = get_linkage(df_sorted_intensities_rec, df_sorted_rts_rec, max_distance = 2 + args.alpha)
	tree = scipy.cluster.hierarchy.to_tree(linkage)

	tree.pre_order(lambda leaf : set_rt(leaf, df_sorted_rts_rec))

	
	leaf_list = sorted(tree.pre_order(lambda leaf : (leaf.id, leaf.rt)), key = lambda tup : tup[0])
	cut_ids = get_cuts(tree, rt_1sWindow = args.rt_1sWindow, frac_peaks = args.frac_peaks)

	clusters = [(df_sorted_intensities_rec.index[p_id], p_id, clust_ids, [df_sorted_intensities_rec.index[i] for i in clust_ids]) for p_id, clust_ids in cut_ids]
	df_cuts = pd.DataFrame(clusters, columns=['peak_name', 'sorted_index', 'cluster_subtree_ids', 'cluster_subtree_names'])

	clusters = []
	for i_p, peak_name in enumerate(df_cuts['peak_name']):
		p_clust = df_cuts['cluster_subtree_names'].values[i_p]	
		clust_size = len(p_clust)
		clust_intensities = []
		clust_rts = []
		clust_mzs = []
		for p in p_clust:
			clust_mzs.append(df.loc[p]['mzmed'])
			clust_intensities.append(df_sorted_intensities.loc[p].to_dict())
			clust_rts.append(df.loc[p]['rtmed'])

		clusters.append((peak_name, p_clust, clust_intensities, clust_mzs, clust_rts, clust_size))
	df_merged = pd.DataFrame(clusters, columns=['parent_peak', 'cluster', 'clust_intensities', 'clust_mzs', 'clust_rts', 'clust_size'])

	clusters = []
	for i_p, parent_peak in enumerate(df_merged['parent_peak']):

		# RT Filtering (only keep peaks that fall within reasonable RT range)
		clust = df_merged['cluster'].values[i_p]
		clust_rts = df_merged['clust_rts'].values[i_p]

		parent_index = df_merged['cluster'].values[i_p].index(parent_peak)
		parent_rt = clust_rts[parent_index]
		lb = parent_rt - args.cluster_outlier_1sWidth
		ub = parent_rt + args.cluster_outlier_1sWidth


		keep_indices = [(r >= lb and r <= ub) for r in clust_rts] 

		dropped = [(i, c) for i, c in enumerate(clust) if not keep_indices[i]]

		clust = [c for i, c in enumerate(clust) if keep_indices[i]]
		clust_intensities = [d for i, d in enumerate(df_merged['clust_intensities'].values[i_p]) if keep_indices[i]]
		clust_mzs = [d for i, d in enumerate(df_merged['clust_mzs'].values[i_p]) if keep_indices[i]]
		clust_rts = [d for i, d in enumerate(clust_rts) if keep_indices[i]]
		clust_size = len(clust)

		clust_avg_intensities = [np.mean(df[args.sample_names].fillna(value = 0.0).loc[x]) for x in clust]
		parent_index = np.argmax(clust_avg_intensities)
		parent_peak = clust[parent_index]
		parent_mz = clust_mzs[parent_index]

		restart = True
		while restart:
			restart = False
			for i_cm, cm in enumerate(clust_mzs):
				if clust_mzs[i_cm] > parent_mz:
					if clust_avg_intensities[i_cm] >= clust_avg_intensities[parent_index] * args.parent_mz_check_intensity_frac:
						parent_index = i_cm
						parent_peak = clust[i_cm]
						parent_mz = clust_mzs[i_cm]
						restart = True
						break

		clusters.append((parent_peak, clust, clust_intensities, clust_mzs, clust_rts, clust_size))

		for i_dpn, dpn in dropped:
			dropped_clust = [dpn]
			dpn_intensities = [df_merged['clust_intensities'].values[i_p][i_dpn]]
			dpn_mzs = [df_merged['clust_mzs'].values[i_p][i_dpn]]
			dpn_rts = [df_merged['clust_rts'].values[i_p][i_dpn]]
			dpn_size = len(dropped_clust)


			clusters.append((dpn, dropped_clust, dpn_intensities, dpn_mzs, dpn_rts, dpn_size))

	df_new_clusters = pd.DataFrame(clusters, columns=['parent_peak', 'cluster', 'clust_intensities', 'clust_mzs', 'clust_rts', 'clust_size'])


	df_clusts_to_merge = df_new_clusters[df_new_clusters['clust_size'] >= 1]
	for i_p, out_parent in enumerate(df_clusts_to_merge['parent_peak']):
		merge_parents = df_clusts_to_merge['cluster'].values[i_p]
		for parent_pn in merge_parents:
			if parent_pn == out_parent:
				continue
			c = df_clusters.loc[parent_pn]
			for i_pn, pn in enumerate(c['cluster']):
				df_clusters.loc[out_parent]['cluster'].append(pn)
				df_clusters.loc[out_parent]['clust_intensities'].append(c['clust_intensities'][i_pn])
				df_clusters.loc[out_parent]['clust_mzs'].append(c['clust_mzs'][i_pn])
				df_clusters.loc[out_parent]['clust_rts'].append(c['clust_rts'][i_pn])
				# df_clusters.loc[out_parent]['clust_size'] = len(df_clusters.loc[out_parent]['cluster'])
				df_clusters.loc[out_parent, 'clust_size'] = len(df_clusters.loc[out_parent]['cluster'])
			df_clusters = df_clusters.drop(parent_pn)
		
	if len(df_clusters) == num_clusters:
		break

	df_clusters.to_csv(os.path.join(args.intermediate_files_dir, args.out_prefix + '_RECURSIVE_ROUND_%i.tsv' % (recursive_round_num)), sep = '\t', index = False)
	num_clusters = len(df_clusters)
	recursive_round_num += 1
	parent_peaks = df_clusters['parent_peak'].values

print('Finished with %i clusters after %i rounds' % (num_clusters, recursive_round_num))
print('Finished with %i clusters after %i rounds' % (num_clusters, recursive_round_num), file = logfile)
sys.stdout.flush()

df_clusters = df_clusters.reset_index(drop=True)
df_clusters['clust_size'] = [len(c) for c in df_clusters['cluster']]


print('done...%s' % str(datetime.now() - time_stamp))
################################################################################
# Normalize peaks
################################################################################
# Normalize the peak intensity values according to the provided 
# normalization file (if one is provided)
################################################################################
df_clusters = df_clusters.set_index('parent_peak', drop = False)
df_clusters.insert(0, 'cluster_id', [i for i,_ in enumerate(df_clusters['cluster'])])
s = df_clusters.apply(lambda x : pd.Series(x['cluster']), axis = 1).stack().reset_index(level=1, drop=True)
s.name = 'name'
df_joined = df_clusters.join(s).drop(labels = ['clust_intensities', 'clust_rts', 'clust_mzs'], axis = 1).set_index('name')

df_normalized = df_joined.merge(df, how = 'outer', left_index = True, right_index = True)

if args.normalization_tsv != None:

	time_stamp = datetime.now()
	print('########################################################################')
	print('Normalizing peaks... %s' % time_stamp)
	print('########################################################################')
	sys.stdout.flush()

	with open(args.normalization_tsv, 'r') as f:
		lines = [x.strip().split('\t') for x in f.readlines()]
	normalization_dict = {x[0] : float(x[1]) for x in lines}

	split_int_cols = True
	if len(set(normalization_dict.keys()).symmetric_difference(set([os.path.splitext(x)[0] for x in args.sample_names]))) > 0:
		if len(set(normalization_dict.keys()).symmetric_difference(set(args.sample_names))) == 0:
			split_int_cols = False
		else:
			print('Normalization file columns: %s' % str(list(normalization_dict.keys())))
			print('args.sample_names columns: %s' % str(args.sample_names))
			sys.exit('Error - normalization file must contain the same columns supplied by args.sample_names')

	for x in args.sample_names:
		if split_int_cols:
			k = os.path.splitext(x)[0]
		else:
			k = x
		df_normalized[x] = df[x] * normalization_dict[k]


	print('done...%s' % str(datetime.now() - time_stamp))

################################################################################
# Fix report values - i.e. log transform etc
################################################################################
# Calculate feature intensity means and rt means for the output report. ALso
# make sure that the order of the samples in the output report corresponds 
# to the order passed in. 
################################################################################
time_stamp = datetime.now()
print('########################################################################')
print('Prettifying output... %s' % time_stamp)
print('########################################################################')
sys.stdout.flush()

df_out = df_normalized.copy()

u_cols = [x for x in df_out.columns if 'Unnamed:' in x]
if len(u_cols) > 0:
	df_out = df_out.drop(labels = u_cols, axis = 1)

clust_size_index = list(df_out.columns).index('clust_size') + 1
df_out.insert(clust_size_index, 'rt_mean', df_sorted_rts.mean(axis = 1).loc[df_out.index])
df_out.insert(clust_size_index, 'intensity_mean', df_out[args.sample_names].fillna(0.0).mean(axis = 1))


insert_index = list(df_out.columns).index('peakidx') + 1 

if args.sample_groups != None:
	sg_means = {}
	for sg in set(args.sample_groups):
		sg_samples = [x for i_x, x in enumerate(args.sample_names) if args.sample_groups[i_x] == sg]
		df_out.insert(insert_index, '%s_mean' % sg, df_out[sg_samples].fillna(0.0).mean(axis = 1))

df_out_temp = pd.DataFrame(df_out[[x for x in df_out.columns if x not in (args.sample_names + args.rt_cols)]])
df_out = pd.concat([df_out_temp, pd.DataFrame(df_out[args.sample_names + args.rt_cols])], axis = 1, sort = False)

mz_cols = [x.replace('rt_', 'mz_') for x in args.rt_cols]
df_out = pd.concat([df_out[[x for x in df_out.columns if x not in mz_cols]], df_out[mz_cols]], axis = 1)
	
print('done...%s' % str(datetime.now() - time_stamp))
################################################################################
# Run statistics
################################################################################
# Calculate cluster statistics if an args.stats_tsv file is passed in. 
# Meant for running the Mann-Whitney test between two conditions
# then for calculating the Q values for the parent peaks (false discovery rate
# correction). The sample names that define the two conditions can be passed in 
# and multiple tests can be run for different condition splits
################################################################################
if args.stats_tsv != None:

	time_stamp = datetime.now()
	print('########################################################################')
	print('Running statistics... %s' % time_stamp)
	print('########################################################################')
	sys.stdout.flush()

	def make_plot(vals, title, fname_out_png, fname_out_svg = None):
		fig, ax = plt.subplots(figsize = (9, 6))
		plt.hist(vals, color = 'lightsteelblue', edgecolor = 'black', bins = 20)
		plt.title(title)
		plt.savefig(fname_out_png, dpi = 400)
		if fname_out_svg != None:
			plt.savefig(fname_out_svg)

	df_tests = pd.read_csv(args.stats_tsv, sep = '\t')

	for (test_name, cols1, cols2, paired) in df_tests.values:
		print(test_name)
		cols1 = cols1.split(',')
		cols2 = cols2.split(',')
		if paired and len(cols1) != len(cols2):
			print(f'Error - stat_test {test_name} calls for paired but doesnt have same number of columns in each group \n\t\t Columns_1 (length {len(cols1)}) - {cols1} \n\t\t Columns_2 (length {len(cols2)}) - {cols2}')
			print('SKIPPING THIS TEST')
			continue
			
		stats = ['not calc' for i in range(len(df))]
		pvals = ['not calc' for i in range(len(df))]
		for i in tqdm(range(len(df_out))):
			if not paired:
				stat, pval = scipy.stats.mannwhitneyu(df_out.iloc[i][cols1].fillna(value = 0.0), df_out.iloc[i][cols2].fillna(value = 0.0), alternative = 'two-sided')
			else:
				stat, pval = scipy.stats.wilcoxon(df_out.iloc[i][cols1].fillna(value = 0.0), df_out.iloc[i][cols2].fillna(value = 0.0), mode = 'exact')
			stats[i] = stat
			pvals[i] = pval
		df_out[f"{test_name}_muStats"] = stats
		df_out[f"{test_name}_muPvals"] =  pvals    
		df_parents = df_out[df_out.apply(lambda x : x.name == x['parent_peak'], axis = 1)]
		qresult = qvalue.qvalue(p = ro.FloatVector(df_parents[f"{test_name}_muPvals"].values))
		df_qresults = pd.DataFrame(np.array(qresult.rx('qvalues')[0]), index = df_parents.index)
		df_out[f"{test_name}_muQvals"] = df_qresults.reindex(df_out.index, fill_value = None).values[:, 0]

		# Make plots for test
		pvc = test_name + '_muPvals'
		qvc = test_name + '_muQvals'
		dfp = df_out[[pvc, qvc]]

		plot_vals = [dfp[pvc], dfp.dropna(subset = [qvc])[pvc], dfp.dropna(subset = [qvc])[qvc]]
		plot_titles = [f"P-values {test_name} all features", f"P-values {test_name} parent features", f"Q-values {test_name} parent features"]
		fnames_png = [os.path.join(args.out_dir, f"{pvc}_allFeats.png"), os.path.join(args.out_dir, f"{pvc}_parentFeats.png"), os.path.join(args.out_dir, f"{qvc}_parentFeats.png")]
		fnames_svg = [os.path.join(args.out_dir, f"{pvc}_allFeats.svg"), os.path.join(args.out_dir, f"{pvc}_parentFeats.svg"), os.path.join(args.out_dir, f"{qvc}_parentFeats.svg")]

		for vals, title, fname_out_png, fname_out_svg in zip(plot_vals, plot_titles, fnames_png, fnames_svg):
			make_plot(vals = vals, title = title, fname_out_png = fname_out_png, fname_out_svg = fname_out_svg)





	data_cols = args.sample_names + args.rt_cols + mz_cols
	df_out = pd.concat([df_out[[x for x in df_out.columns if x not in data_cols]], df_out[data_cols]], axis = 1)


	print('done...%s' % str(datetime.now() - time_stamp))


################################################################################
# Wrap up
################################################################################
time_stamp = datetime.now()
print('########################################################################')
print('Writing output... %s' % time_stamp)
print('########################################################################')
sys.stdout.flush()



time_stamp = datetime.now()
print('Reordering final df and saving..%s.' % time_stamp)
sys.stdout.flush()

df_out = df_out.reindex(dend_order, axis = 0)
df_out.to_csv(args.out_tsv, sep = '\t')

print('done...%s' % str(datetime.now() - time_stamp))


print('Completed %s' % str(datetime.now()))
sys.stdout.flush()

logfile.close()

print('done! %s' % str(datetime.now() - time_stamp))




