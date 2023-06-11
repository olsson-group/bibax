import pandas as pd
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#Load data
DMS = pd.read_csv("Deep_Mutational_Scan_RBD_stability_ACE2binding.csv")
dms2 = pd.read_csv("Tableofmutation_antibody-escape_fraction_scores.csv")

# Identify missing sites
missing_sites = np.setdiff1d(np.unique(DMS['site_SARS2'].to_numpy()), np.unique(dms2['site'].to_numpy()), )
print(f"Number of sites missing nAb data, {len(missing_sites)}: "+ str(missing_sites))

missing_sites2 = np.setdiff1d(np.unique(dms2['site'].to_numpy()), np.unique(DMS['site_SARS2'].to_numpy()) )
print(f"Number of sites missing ACE2 data, {len(missing_sites2)}: "+ str(missing_sites2))

# Primary and secondary interfaces determined based on nAb RBD crystal structures:
# 6XC2, 6XCN,6XDG, 6XE1, 6XKQ,7CDI, 7CDJ, 7JMO, 7JMP,7JMW,7JV2,7JV6,7JVA,7JW0,7K90,7LX5.

from itertools import chain
interface_residues = [(443, 463), (469, 507)]
secondary_interface_residues = [(403, 410), (417, 423)]
total_interface_range = list(chain(*[range(*r) for r in interface_residues+secondary_interface_residues]))


# pre-proccessing 
antibody_condition, mutation, escape = (dms2['condition'].to_numpy(),dms2['wildtype'].to_numpy()+dms2['site'].to_numpy().astype('str')+dms2['mutation'].to_numpy(), dms2['mut_escape'].to_numpy())
unique_abc = np.unique(antibody_condition)
all_mutations = DMS['mutation'].to_numpy()
all_expression = np.zeros(len(all_mutations)*len(unique_abc))
all_ace_binding = np.zeros(len(all_mutations)*len(unique_abc))
all_escape = np.zeros(len(all_mutations)*len(unique_abc))

j=0
for abc in unique_abc:
    idx_condition = antibody_condition==abc
    for i,m in enumerate(all_mutations):
        candidate=DMS.loc[np.where(DMS['mutation'].to_numpy()==m)[0]]
        all_expression[i+j*len(all_mutations)] = candidate['expr_avg'].to_numpy()
        all_ace_binding[i+j*len(all_mutations)] = candidate['bind_avg'].to_numpy()
        condition_selection = np.where(((mutation==m)*(idx_condition)))[0]
        if len(condition_selection)>0:
            all_escape[i+j*len(all_mutations)] = escape[condition_selection[0]]
        else:
            all_escape[i+j*len(all_mutations)] = np.nan
    
    j=j+1


finite_values = ~(~np.isfinite(all_expression)+~np.isfinite(all_ace_binding))

all_expression = all_expression[finite_values]
all_ace_binding = all_ace_binding[finite_values]
all_mutations = np.tile(all_mutations, len(unique_abc))[finite_values]
all_resid = np.array([int(a[1:-1]) for a in all_mutations])
all_escape = all_escape[finite_values]
_all_mutations = DMS['mutation'].to_numpy()

all_conditions = np.concatenate([np.tile(abc, len(_all_mutations)) for abc in unique_abc ])[finite_values]

def standardize(x):
    xhat=(x-x[np.isfinite(x)].mean())
    return xhat/xhat[np.isfinite(xhat)].std()

def compute_score(expression, escape, binding):
    escape_finite = np.isfinite(escape)
    escape_impute = np.array(escape)
    escape_impute[~escape_finite] = 0.
    return expression - binding - escape_impute

#enforce constraints 
alpha=0.5
beta=1.0
constraints_fulfilled = ((all_expression>-alpha).astype(int)*(all_ace_binding<-beta).astype(int)).astype(bool)

#compute score on data fulfilling constraints
tot_score=compute_score(standardize(all_expression[constraints_fulfilled]), standardize(all_escape[constraints_fulfilled]), standardize(all_ace_binding[constraints_fulfilled]))


#identify hotspots
mutation_hotspots = {}
for i in np.argsort(tot_score)[::-1]:
    y = all_mutations[constraints_fulfilled][i]
    resi = all_resid[constraints_fulfilled][i]
    if int(y[1:-1]) in total_interface_range:
        if mutation_hotspots.get(resi):
            mutation_hotspots[resi].append([y, all_expression[constraints_fulfilled][i], all_ace_binding[constraints_fulfilled][i], tot_score[i]])
        else:
            mutation_hotspots[resi]=[[y, all_expression[constraints_fulfilled][i], all_ace_binding[constraints_fulfilled][i], tot_score[i]]]


# Filtering of low-scoring variants
max_total_scores = np.array([np.max([l[-1] for l in mutation_hotspots[key]]) for key in mutation_hotspots])
argsort_total_scores = np.argsort(max_total_scores)[::-1] # decending scores by index
positive_max_total_scores = max_total_scores[argsort_total_scores]>0.
scores_positive_and_sorted = max_total_scores[argsort_total_scores][positive_max_total_scores]

gamma = 0.40

bisect_res = 1000

score_thres = np.linspace(0,scores_positive_and_sorted[0],bisect_res)[np.argmin([(gamma-np.mean(threshold>scores_positive_and_sorted))**2. for threshold in np.linspace(0,scores_positive_and_sorted[0],bisect_res)])]

print(f"score threshold for gamma={gamma}: ", )

# Diagnostic plot
plt.plot(np.linspace(0,scores_positive_and_sorted[0],bisect_res), [np.mean(threshold>scores_positive_and_sorted) for threshold in np.linspace(0,scores_positive_and_sorted[0],bisect_res)])
plt.vlines([score_thres],ymin=0, ymax=[gamma],color='k')
plt.hlines([gamma],xmin=0, xmax=[score_thres],color='k')
plt.xlabel("score threshold")
plt.ylabel(r"$\gamma$")

def stringify_top_mutants(mutation_hotspots, key, site_variant_scores):
    sorted_hits = np.argsort(site_variants_scores)[::-1]
    hits, counts = np.unique([mutation_hotspots[key][yy][0] for yy in  sorted_hits], return_counts=True)
    return ', '.join([hit+f"({count})" for hit,count in zip(hits,counts) ])

top_mutants=[]
for i,key in enumerate(mutation_hotspots.keys()):
    if max_total_scores[i]>score_thres:
        site_variants_scores = [variant[-1] for variant in mutation_hotspots[key]]
        print(key,'\t', stringify_top_mutants(mutation_hotspots, key, site_variants_scores),'\t', np.round(max_total_scores[i],2))
        top_mutants.append(mutation_hotspots[key][np.argmax(site_variants_scores)][0])

# generate plot summarizing selected variants in the context of all data and constraints.
fig, ax = plt.subplots(4,1, figsize=(4,12), gridspec_kw={'height_ratios': [2,2,1, 1]},constrained_layout=True )
ax[0].scatter(dms2['site'].to_numpy(), dms2['site_max_escape'].to_numpy(), marker='.',alpha=0.1, label="Site for escape mutations")
for i,ir in enumerate(interface_residues):
    if i==0:
        ax[0].hlines(1.05, *ir, color='k', lw=5, label="primary nAb interface")
    else:
        ax[0].hlines(1.05, *ir, color='k', lw=5)
for i,ir in enumerate(secondary_interface_residues):
    if i==0:
        ax[0].hlines(1.05, *ir, color='r', lw=5, label="secondary nAb interface")
    else:
        ax[0].hlines(1.05, *ir, color='r', lw=5)

ax[0].vlines(missing_sites, ymin=-0.08, ymax=0.0, color='r',lw=0.7, label="missing site" )

ax[0].set_xlabel("RBD position")
ax[0].set_ylabel("Maximum Escape")
ax[0].legend()
ax[1].scatter(DMS['expr_avg'].to_numpy(),
     DMS['bind_avg'].to_numpy(),marker='.',c='k',alpha=0.5)

ax[1].scatter(all_expression[constraints_fulfilled][np.isin(all_mutations[constraints_fulfilled], top_mutants)], all_ace_binding[constraints_fulfilled][np.isin(all_mutations[constraints_fulfilled], top_mutants)],marker='o',c='r',alpha=0.8)


ax[1].set_xlabel("Expression")
ax[1].set_ylabel("ACE2 binding")
rect = patches.Rectangle((-0.5, -5), 1.5, 4.00, linewidth=0, edgecolor='none', facecolor='y', alpha=0.2)

ax[1].add_patch(rect)

ax[3].scatter(DMS['expr_avg'].to_numpy(),
     DMS['bind_avg'].to_numpy(),marker='.',c='k',alpha=0.5)

ax[3].scatter(all_expression[constraints_fulfilled][np.isin(all_mutations[constraints_fulfilled], top_mutants)], all_ace_binding[constraints_fulfilled][np.isin(all_mutations[constraints_fulfilled], top_mutants)],marker='o',c='r',alpha=0.8)

ax[2].scatter(DMS['expr_avg'].to_numpy(),
     DMS['bind_avg'].to_numpy(),marker='.',c='k',alpha=0.5)

ax[2].scatter(all_expression[constraints_fulfilled][np.isin(all_mutations[constraints_fulfilled], top_mutants)], all_ace_binding[constraints_fulfilled][np.isin(all_mutations[constraints_fulfilled], top_mutants)],marker='o',c='r',alpha=0.8)
ax[2].set_xlim(-0.5,1)
ax[2].set_ylim(-2.1,-1.2)
ax[2].set_xticklabels([])

ax[3].set_xlim(-0.5,1)
ax[3].set_ylim(-4.85,-4.6)
ax[3].set_xlabel("Expression")
ax[3].set_ylabel("ACE2 binding")
ax[2].set_ylabel("ACE2 binding")


for bm in top_mutants:
    _idx = (DMS['mutation']==str(bm)).to_numpy().argmax()
    ax[2].annotate(str(bm), (DMS['expr_avg'][_idx], DMS['bind_avg'][_idx] ), va='center', ha='center')

for bm in top_mutants:
    _idx = (DMS['mutation']==str(bm)).to_numpy().argmax()
    ax[3].annotate(str(bm), (DMS['expr_avg'][_idx], DMS['bind_avg'][_idx] ), va='center', ha='center')


for _,let in zip(ax, ["A","B","C"]):
    _.text(-0.2,1.05, let, fontsize=16, transform=_.transAxes)
#plt.tight_layout()
plt.savefig("nAb_RBD_ACE2_DMS_Bloom.pdf")



RBD_WT_AA_Seq=''.join(DMS['wildtype'].to_numpy().reshape(-1,21)[:,0])


# algorithms and functions to generate csv of top-ranked variants

dna2aa = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}


aa2dna = { #la
    dna2aa[key]:key for key in dna2aa.keys()
}

def translate(seq):
    prot=""
    for i in range(0,len(seq),3):
        prot=prot+dna2aa[seq[i:i+3]]
    return prot

# reference sites

RBD_start = 954
RBD_start_Bloom = 954+36 
RBD_end = 1623
RBD_end_Bloom = 1623-30

# `wildtype` nucleotide sequence
spike_reference_sequence="ATGTTTGTCTTCCTGGTCCTGCTGCCTCTGGTCTCCTCTCAGTGCGTGAACCTGACTACTAGAACTCAGCTGCCTCCCGCTTACACCAATAGCTTCACCAGGGGCGTGTACTATCCAGACAAGGTGTTTCGCAGCTCCGTGCTGCACTCCACACAGGATCTGTTTCTGCCCTTCTTTTCTAACGTGACCTGGTTCCACGCCATCCACGTGTCCGGCACCAATGGCACAAAGAGGTTCGACAATCCAGTGCTGCCCTTTAACGATGGCGTGTACTTCGCCTCCACCGAGAAGTCTAACATCATCCGCGGCTGGATCTTTGGCACCACACTGGACAGCAAGACACAGTCCCTGCTGATCGTGAACAATGCCACCAACGTGGTCATCAAGGTGTGCGAGTTCCAGTTTTGTAATGATCCTTTCCTGGGCGTGTACTATCACAAGAACAATAAGTCTTGGATGGAGAGCGAGTTTAGGGTGTACTCTAGCGCCAACAATTGCACATTTGAGTATGTGAGCCAGCCATTCCTGATGGACCTGGAGGGCAAGCAGGGCAATTTCAAGAACCTGCGGGAGTTCGTGTTTAAGAATATCGATGGCTACTTCAAAATCTACTCCAAGCACACCCCCATCAACCTGGTGCGGGACCTGCCACAGGGCTTCTCTGCCCTGGAGCCTCTGGTGGATCTGCCAATCGGCATCAACATCACCAGGTTTCAGACACTGCTGGCCCTGCACCGCAGCTACCTGACACCTGGCGACTCCTCTAGCGGATGGACCGCAGGAGCTGCCGCCTACTATGTGGGCTACCTGCAGCCAAGGACCTTCCTGCTGAAGTATAACGAGAATGGCACCATCACAGACGCAGTGGATTGCGCCCTGGACCCCCTGTCTGAGACAAAGTGTACACTGAAGAGCTTTACCGTGGAGAAGGGCATCTACCAGACAAGCAATTTCCGGGTGCAGCCCACCGAGTCCATCGTGAGATTTCCAAATATCACAAACCTGTGCCCCTTTGGCGAGGTGTTCAACGCCACCCGCTTCGCCAGCGTGTATGCCTGGAATAGGAAGCGCATCTCCAACTGCGTGGCCGACTATTCTGTGCTGTACAACTCCGCCTCTTTCAGCACCTTTAAGTGTTACGGCGTGAGCCCTACAAAGCTGAATGACCTGTGCTTTACCAACGTGTATGCCGATTCCTTCGTGATCAGGGGCGACGAGGTGCGCCAGATCGCACCAGGACAGACAGGCAAGATCGCCGACTACAATTATAAGCTGCCCGACGATTTCACCGGCTGCGTGATCGCCTGGAACTCTAACAATCTGGATAGCAAAGTGGGCGGCAACTACAATTATCTGTACCGGCTGTTTAGAAAGTCTAATCTGAAGCCTTTCGAGAGGGACATCTCCACAGAAATCTACCAGGCCGGCTCTACCCCATGCAATGGCGTGGAGGGCTTTAACTGTTATTTCCCCCTGCAGTCCTACGGCTTCCAGCCTACAAACGGCGTGGGCTATCAGCCATACCGCGTGGTGGTGCTGTCTTTTGAGCTGCTGCACGCACCAGCAACAGTGTGCGGACCTAAGAAGAGCACCAATCTGGTGAAGAACAAGTGCGTGAACTTCAACTTCAACGGCCTGACCGGCACAGGCGTGCTGACCGAGAGCAACAAGAAGTTCCTGCCCTTTCAGCAGTTCGGCCGGGACATCGCAGATACCACAGACGCCGTGCGGGACCCCCAGACCCTGGAGATCCTGGACATCACACCTTGCTCCTTCGGCGGCGTGTCTGTGATCACACCTGGCACCAATACATCCAACCAGGTGGCCGTGCTGTACCAGGACGTGAATTGTACCGAGGTGCCAGTGGCCATCCACGCCGATCAGCTGACCCCCACATGGAGGGTGTATAGCACCGGCTCCAACGTGTTCCAGACACGCGCCGGATGCCTGATCGGAGCAGAGCACGTGAACAATAGCTACGAGTGCGACATCCCCATCGGCGCCGGCATCTGTGCCTCCTATCAGACCCAGACAAACTCCCCTAGGAGAGCCCGGTCTGTGGCCTCCCAGTCTATCATCGCCTACACCATGAGCCTGGGCGCCGAGAACAGCGTGGCCTATTCTAACAATAGCATCGCCATCCCCACCAACTTCACAATCAGCGTGACCACAGAGATCCTGCCTGTGAGCATGACCAAGACATCCGTGGACTGCACAATGTACATCTGTGGCGATTCCACCGAGTGCTCTAACCTGCTGCTGCAGTATGGCTCCTTTTGTACCCAGCTGAATAGAGCCCTGACAGGCATCGCCGTGGAGCAGGACAAGAACACACAGGAGGTGTTCGCCCAGGTGAAGCAAATCTACAAGACCCCCCCTATCAAGGACTTTGGCGGCTTCAACTTCAGCCAGATCCTGCCCGATCCTTCCAAGCCATCTAAGCGGAGCTTTATCGAGGACCTGCTGTTCAACAAGGTGACCCTGGCCGATGCCGGCTTCATCAAGCAGTACGGCGATTGCCTGGGCGACATCGCAGCCCGGGACCTGATCTGCGCCCAGAAGTTTAATGGCCTGACCGTGCTGCCACCCCTGCTGACAGATGAGATGATCGCCCAGTATACATCTGCCCTGCTGGCCGGCACCATCACAAGCGGATGGACCTTCGGCGCAGGAGCCGCCCTGCAGATCCCCTTTGCCATGCAGATGGCCTACAGATTCAACGGCATCGGCGTGACCCAGAATGTGCTGTATGAGAACCAGAAGCTGATCGCCAATCAGTTTAACAGCGCCATCGGCAAGATCCAGGACTCTCTGTCCTCTACAGCCAGCGCCCTGGGCAAGCTGCAGGATGTGGTGAATCAGAACGCCCAGGCCCTGAATACCCTGGTGAAGCAGCTGAGCAGCAACTTCGGCGCCATCTCTAGCGTGCTGAATGACATCCTGAGCCGGCTGGACAAAGTTGAGGCAGAGGTGCAGATCGACCGGCTGATCACAGGCAGACTGCAGTCCCTGCAGACCTACGTGACACAGCAGCTGATCAGGGCAGCAGAGATCAGGGCCTCTGCCAATCTGGCCGCCACCAAGATGAGCGAGTGCGTGCTGGGCCAGTCCAAGAGAGTGGACTTTTGTGGCAAGGGCTACCACCTGATGAGCTTCCCACAGTCCGCCCCCCACGGCGTGGTGTTTCTGCACGTGACCTATGTGCCTGCCCAGGAGAAGAACTTCACCACAGCCCCAGCCATCTGCCACGATGGCAAGGCACACTTTCCCCGGGAGGGCGTGTTCGTGAGCAACGGCACCCACTGGTTTGTGACACAGAGAAATTTCTATGAGCCTCAGATCATCACCACAGACAATACCTTCGTGAGCGGCAACTGTGACGTGGTCATCGGCATCGTGAACAATACCGTGTACGATCCTCTGCAGCCAGAGCTGGACTCTTTTAAGGAGGAGCTGGATAAGTATTTCAAGAACCACACCAGCCCCGACGTGGATCTGGGCGACATCTCTGGCATCAATGCCAGCGTGGTGAACATCCAGAAGGAGATCGACAGGCTGAATGAGGTGGCCAAGAATCTGAACGAGAGCCTGATCGATCTGCAGGAGCTGGGCAAGTATGAGCAGTCCGGCCGCGAGAACCTGTACTTCCAGGGAGGAGGAGGCTCTGGATATATCCCAGAGGCACCTCGGGATGGACAGGCCTACGTGAGAAAAGACGGCGAGTGGGTCCTGCTGAGTACCTTCCTGGGGCATCATCATCATCACCATTAA"
# `2-P variant` nucleotide sequence
spike_reference_sequence_2p="ATGTTTGTCTTCCTGGTCCTGCTGCCTCTGGTCTCCTCTCAGTGCGTGAACCTGACTACTAGAACTCAGCTGCCTCCCGCTTACACCAATAGCTTCACCAGGGGCGTGTACTATCCAGACAAGGTGTTTCGCAGCTCCGTGCTGCACTCCACACAGGATCTGTTTCTGCCCTTCTTTTCTAACGTGACCTGGTTCCACGCCATCCACGTGTCCGGCACCAATGGCACAAAGAGGTTCGACAATCCAGTGCTGCCCTTTAACGATGGCGTGTACTTCGCCTCCACCGAGAAGTCTAACATCATCCGCGGCTGGATCTTTGGCACCACACTGGACAGCAAGACACAGTCCCTGCTGATCGTGAACAATGCCACCAACGTGGTCATCAAGGTGTGCGAGTTCCAGTTTTGTAATGATCCTTTCCTGGGCGTGTACTATCACAAGAACAATAAGTCTTGGATGGAGAGCGAGTTTAGGGTGTACTCTAGCGCCAACAATTGCACATTTGAGTATGTGAGCCAGCCATTCCTGATGGACCTGGAGGGCAAGCAGGGCAATTTCAAGAACCTGCGGGAGTTCGTGTTTAAGAATATCGATGGCTACTTCAAAATCTACTCCAAGCACACCCCCATCAACCTGGTGCGGGACCTGCCACAGGGCTTCTCTGCCCTGGAGCCTCTGGTGGATCTGCCAATCGGCATCAACATCACCAGGTTTCAGACACTGCTGGCCCTGCACCGCAGCTACCTGACACCTGGCGACTCCTCTAGCGGATGGACCGCAGGAGCTGCCGCCTACTATGTGGGCTACCTGCAGCCAAGGACCTTCCTGCTGAAGTATAACGAGAATGGCACCATCACAGACGCAGTGGATTGCGCCCTGGACCCCCTGTCTGAGACAAAGTGTACACTGAAGAGCTTTACCGTGGAGAAGGGCATCTACCAGACAAGCAATTTCCGGGTGCAGCCCACCGAGTCCATCGTGAGATTTCCAAATATCACAAACCTGTGCCCCTTTGGCGAGGTGTTCAACGCCACCCGCTTCGCCAGCGTGTATGCCTGGAATAGGAAGCGCATCTCCAACTGCGTGGCCGACTATTCTGTGCTGTACAACTCCGCCTCTTTCAGCACCTTTAAGTGTTACGGCGTGAGCCCTACAAAGCTGAATGACCTGTGCTTTACCAACGTGTATGCCGATTCCTTCGTGATCAGGGGCGACGAGGTGCGCCAGATCGCACCAGGACAGACAGGCAAGATCGCCGACTACAATTATAAGCTGCCCGACGATTTCACCGGCTGCGTGATCGCCTGGAACTCTAACAATCTGGATAGCAAAGTGGGCGGCAACTACAATTATCTGTACCGGCTGTTTAGAAAGTCTAATCTGAAGCCTTTCGAGAGGGACATCTCCACAGAAATCTACCAGGCCGGCTCTACCCCATGCAATGGCGTGGAGGGCTTTAACTGTTATTTCCCCCTGCAGTCCTACGGCTTCCAGCCTACAAACGGCGTGGGCTATCAGCCATACCGCGTGGTGGTGCTGTCTTTTGAGCTGCTGCACGCACCAGCAACAGTGTGCGGACCTAAGAAGAGCACCAATCTGGTGAAGAACAAGTGCGTGAACTTCAACTTCAACGGCCTGACCGGCACAGGCGTGCTGACCGAGAGCAACAAGAAGTTCCTGCCCTTTCAGCAGTTCGGCCGGGACATCGCAGATACCACAGACGCCGTGCGGGACCCCCAGACCCTGGAGATCCTGGACATCACACCTTGCTCCTTCGGCGGCGTGTCTGTGATCACACCTGGCACCAATACATCCAACCAGGTGGCCGTGCTGTACCAGGACGTGAATTGTACCGAGGTGCCAGTGGCCATCCACGCCGATCAGCTGACCCCCACATGGAGGGTGTATAGCACCGGCTCCAACGTGTTCCAGACACGCGCCGGATGCCTGATCGGAGCAGAGCACGTGAACAATAGCTACGAGTGCGACATCCCCATCGGCGCCGGCATCTGTGCCTCCTATCAGACCCAGACAAACTCCCCTAGGAGAGCCCGGTCTGTGGCCTCCCAGTCTATCATCGCCTACACCATGAGCCTGGGCGCCGAGAACAGCGTGGCCTATTCTAACAATAGCATCGCCATCCCCACCAACTTCACAATCAGCGTGACCACAGAGATCCTGCCTGTGAGCATGACCAAGACATCCGTGGACTGCACAATGTACATCTGTGGCGATTCCACCGAGTGCTCTAACCTGCTGCTGCAGTATGGCTCCTTTTGTACCCAGCTGAATAGAGCCCTGACAGGCATCGCCGTGGAGCAGGACAAGAACACACAGGAGGTGTTCGCCCAGGTGAAGCAAATCTACAAGACCCCCCCTATCAAGGACTTTGGCGGCTTCAACTTCAGCCAGATCCTGCCCGATCCTTCCAAGCCATCTAAGCGGAGCTTTATCGAGGACCTGCTGTTCAACAAGGTGACCCTGGCCGATGCCGGCTTCATCAAGCAGTACGGCGATTGCCTGGGCGACATCGCAGCCCGGGACCTGATCTGCGCCCAGAAGTTTAATGGCCTGACCGTGCTGCCACCCCTGCTGACAGATGAGATGATCGCCCAGTATACATCTGCCCTGCTGGCCGGCACCATCACAAGCGGATGGACCTTCGGCGCAGGAGCCGCCCTGCAGATCCCCTTTGCCATGCAGATGGCCTACAGATTCAACGGCATCGGCGTGACCCAGAATGTGCTGTATGAGAACCAGAAGCTGATCGCCAATCAGTTTAACAGCGCCATCGGCAAGATCCAGGACTCTCTGTCCTCTACAGCCAGCGCCCTGGGCAAGCTGCAGGATGTGGTGAATCAGAACGCCCAGGCCCTGAATACCCTGGTGAAGCAGCTGAGCAGCAACTTCGGCGCCATCTCTAGCGTGCTGAATGACATCCTGAGCCGGCTGGACCCCCCAGAGGCAGAGGTGCAGATCGACCGGCTGATCACAGGCAGACTGCAGTCCCTGCAGACCTACGTGACACAGCAGCTGATCAGGGCAGCAGAGATCAGGGCCTCTGCCAATCTGGCCGCCACCAAGATGAGCGAGTGCGTGCTGGGCCAGTCCAAGAGAGTGGACTTTTGTGGCAAGGGCTACCACCTGATGAGCTTCCCACAGTCCGCCCCCCACGGCGTGGTGTTTCTGCACGTGACCTATGTGCCTGCCCAGGAGAAGAACTTCACCACAGCCCCAGCCATCTGCCACGATGGCAAGGCACACTTTCCCCGGGAGGGCGTGTTCGTGAGCAACGGCACCCACTGGTTTGTGACACAGAGAAATTTCTATGAGCCTCAGATCATCACCACAGACAATACCTTCGTGAGCGGCAACTGTGACGTGGTCATCGGCATCGTGAACAATACCGTGTACGATCCTCTGCAGCCAGAGCTGGACTCTTTTAAGGAGGAGCTGGATAAGTATTTCAAGAACCACACCAGCCCCGACGTGGATCTGGGCGACATCTCTGGCATCAATGCCAGCGTGGTGAACATCCAGAAGGAGATCGACAGGCTGAATGAGGTGGCCAAGAATCTGAACGAGAGCCTGATCGATCTGCAGGAGCTGGGCAAGTATGAGCAGTCCGGCCGCGAGAACCTGTACTTCCAGGGAGGAGGAGGCTCTGGATATATCCCAGAGGCACCTCGGGATGGACAGGCCTACGTGAGAAAAGACGGCGAGTGGGTCCTGCTGAGTACCTTCCTGGGGCATCATCATCATCACCATTAA"


# Default output:
# 1. Nucleotide sequence RBD
# 2. Nucleotide sequence full spike (with 2-P mutation)
# 3. Aminoacid sequence RBD
# 4. Aminoacid sequence full spike (with 2-P mutation)
# 
# return 2 dna seqs, 2 prot seqs
#


def build_sequences(wt_dna_full_spike, wt_dna_full_spike_2p, rbd_variant):
    #parse mutation
    origin_aa, site, target_aa = rbd_variant[0],int(rbd_variant[1:-1]), rbd_variant[-1]
    site_to_dna = (site-1)*3
    target_codon = aa2dna[target_aa]
    #apply mutation
    mutated_seq_ = wt_dna_full_spike[:site_to_dna]+target_codon+wt_dna_full_spike[site_to_dna+3:]
    mutated_seq_2p = wt_dna_full_spike_2p[:site_to_dna]+target_codon+wt_dna_full_spike_2p[site_to_dna+3:]
    #translate sequences
    translated_sequence = translate(mutated_seq_[RBD_start:RBD_end])
    translated_sequence_2p = translate(mutated_seq_2p)
    return {"Variant_Nucleic_Acid_sequence_RBD": mutated_seq_[RBD_start:RBD_end], "Variant_Nucleic_Acid_sequence_2-P":mutated_seq_2p, "Variant_Amino_Acid_sequence_RBD":translated_sequence, "Variant_Amino_Acid_sequence_2-P":translated_sequence_2p[:-1]}

def aggregate(mut, scor, spike_reference=None, spike_reference_2p=None, thres=0):
    out = {}
    sequences = {}
    for m, s in zip(mut, scor):
        if s>thres:
            if out.get(m):
                out[m].append(s)
            else:
                out[m]=[s]
                _seqs = build_sequences(spike_reference,spike_reference_2p, m)
                for key in _seqs.keys():
                    if sequences.get(key):
                        sequences[key].append(_seqs[key])
                    else:
                        sequences[key]=[_seqs[key]]

    for key in out.keys():
        out[key] = np.round(np.mean(out[key]),3)
    
    return {**{"mutant":list(out.keys()), "scores":[out[s] for s in out.keys()]}, **sequences}

decending_sort_tot_score = np.argsort(tot_score)[::-1]
mutants = all_mutations[constraints_fulfilled][decending_sort_tot_score]
full_ranking=pd.DataFrame(aggregate(mutants, tot_score[decending_sort_tot_score], spike_reference=spike_reference_sequence, spike_reference_2p=spike_reference_sequence_2p, thres=score_thres))
#full_ranking.to_csv("top-variant-ranking_2.csv")