import os
import re
import sys
import glob
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib
import seaborn as sns
from matplotlib_venn import venn2, venn2_unweighted
import matplotlib.pyplot as plt
from pyfaidx import Fasta

genome = Fasta('/juno/work/shah/users/chois7/mmctm/reference/GRCh37-lite.fa')
isabl_maf_cols = ['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2',]
tempo_maf_cols = isabl_maf_cols 
dst_maf_cols = ['chrom', 'pos', 'ref', 'alt']


def plot_mutation_spectra(snv, fig, ax, tag, despine=False):
    font = matplotlib.font_manager.FontProperties()
    font.set_family('monospace')

    df = snv.copy()
    pat = r'([ACGT])\[([ACGT])\>([ACGT])\]([ACGT])'
    df['norm_tri_nt'] = df.index.str.replace(pat, r'\1\2\4', regex=True) 
    df['norm_mut_type'] = df.index.str.replace(pat, r'\2>\3', regex=True) 
    df['index'] = range(df.shape[0])
    
    for mut_type, mut_type_data in df.groupby(['norm_mut_type']):
        ax.bar(data=mut_type_data, x='index', height='probability', label=mut_type)
        
    ax.set_xticks(df['index'])
    ax.set_xticklabels(df['norm_tri_nt'], rotation=90, fontproperties=font)
    ax.set_xlim((-1, 97))
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_title(tag)
    
    sns.despine(ax=ax, trim=True)

    
def plot_snv_terms(title, snv, save_path=None, tag=''):
    fig, ax = plt.subplots(1)
    fig.set_figheight(3)
    fig.set_figwidth(20)
    
    if tag: title += tag
    fig.suptitle(title)
    
    plot_mutation_spectra(snv, fig, ax, tag)
    plt.tight_layout()
    
    if save_path: plt.savefig(save_path)
    
    
def plot_paired_snv_terms(title, snv1, snv2, save_path=None, tags=None):
    fig, ax = plt.subplots(2)
    ax1, ax2 = ax
    fig.set_figheight(7)
    fig.set_figwidth(20)
    
    fig.suptitle(title)
    
    tag1, tag2 = tags
    plot_mutation_spectra(snv1, fig, ax1, tag1)
    plot_mutation_spectra(snv2, fig, ax2, tag2)
    plt.tight_layout()
    
    if save_path: plt.savefig(save_path)

def construct_empty_count_series():
    snvs = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    nts = ['A', 'C', 'G', 'T']
    terms = [
        '{}[{}]{}'.format(l, s, r) for s in snvs for l in nts for r in nts
    ]
    return pd.Series(np.zeros(len(terms), dtype=int), index=terms)


def normalize_snv(context, alt):
    ref = context.seq[1]

    if ref in ['A', 'G']:
        context = (-context).seq

        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
        alt = complement[str(alt)]
    else:
        context = context.seq
        alt = str(alt)
    return context, alt


def construct_snv_label(context, alt):
    if len(context) != 3:
        warnings.warn('Warning: bad context length: {}'.format(str(context)))
        return None
    return '{}[{}>{}]{}'.format(context[0], context[1], alt, context[2])

def count_snvs(snvs, genome):
    df = snvs.copy()

    var_converter = {
        'G>T': 'C>A', 
        'G>C': 'C>G', 
        'G>A': 'C>T', 
        'A>T': 'T>A', 
        'A>G': 'T>C', 
        'A>C': 'T>G'
    }
    df['var_type'] = (df['ref'] + '>' + df['alt']).replace(var_converter)
    
    counts = construct_empty_count_series()
    contexts = []

    for idx, row in snvs.iterrows():
        # two flanking bases
        start = row['pos'] - 2
        end = row['pos'] + 1

        context = genome[row['chrom']][start:end]
        if 'N' in context.seq:
            warnings.warn(
                'Warning: N found in context sequence at {}:{}-{}'.format(
                    row['chrom'], start + 1, end
                )
            )
            continue

        context, alt = normalize_snv(context, row['alt'])
        counts[construct_snv_label(context, alt)] += 1
        contexts.append(context)
    df['context'] = pd.Series(contexts)
    df['var'] = df['context'].str[0] + '[' + df['var_type'] + ']' + df['context'].str[-1]

    return counts, df

def make_var_set(data, vartype='snv'):
    if vartype == 'snv':
        ix_cols = ['chrom', 'pos', 'ref', 'alt']
    elif vartype == 'sv':
        ix_cols = ['chromosome_1', 'position_1', 'strand_1', 'chromosome_2', 'position_2', 'strand_2']
    df_ix = data.copy().set_index(ix_cols).index.tolist()
    df_ix_set = set(df_ix)
    # assert len(df_ix_set) == len(df_ix)
    return df_ix_set

def get_tempo_snv(snv_path, tempo_maf_cols=tempo_maf_cols, dst_maf_cols=dst_maf_cols):
    snv = pd.read_table(snv_path, comment='#', dtype={'Chromosome': str}, low_memory=False)
    snv = snv[snv['Variant_Type'] == 'SNP']
    if tempo_maf_cols and dst_maf_cols:
        snv.rename(columns=dict(zip(tempo_maf_cols, dst_maf_cols)), inplace=True)
    return snv

def print_set_stats(consensus_match, consensus_nonmatch, tempo_match, tempo_nonmatch,
                  isabl_tumor_id, tempo_tumor_id, isabl_normal_id, tempo_normal_id, vartype='snv'):
    A_and_B = make_var_set(consensus_match, vartype=vartype)
    A_and_B_from_tempo = make_var_set(tempo_match, vartype=vartype)
    assert abs(len(A_and_B) - len(A_and_B_from_tempo)) <= 10, f'A_and_B:\n{A_and_B}\nA_and_B_from_tempo:\n{A_and_B_from_tempo}\n'
    A_not_B = make_var_set(consensus_nonmatch, vartype=vartype)
    B_not_A = make_var_set(tempo_nonmatch, vartype=vartype)
    
    A = A_and_B.union(A_not_B)
    B = A_and_B.union(B_not_A)
    
    field = [isabl_tumor_id, tempo_tumor_id, isabl_normal_id, tempo_normal_id, 
             len(A), len(B), len(A-B), len(B-A), len(A&B), len(A|B)]
    field = [str(_) for _ in field]
    return A, B, field # for venn2 plott

def map_tempo_snv_by_id(tempo_snv_data, isabl_tumor_id, isabl_normal_id, dst_maf_cols=dst_maf_cols):
    assert isabl_tumor_id in tempo_snv_data
    tempo_normal_id, tempo_snv_path = tempo_snv_data[isabl_tumor_id]
    tempo_tumor_id = isabl_tumor_id
    if isabl_normal_id != tempo_normal_id:
         print(f"WARNING: isabl ({isabl_normal_id}) != tempo ({tempo_normal_id})", file=sys.stderr)
    tempo = get_tempo_snv(tempo_snv_path)
    tempo = tempo[dst_maf_cols]
        
    return tempo, tempo_tumor_id, tempo_normal_id

def extract_isabl_normal_id(wgsmetadata, isabl_tumor_id):    
    """Return normal ID from .err"""
    wgsmeta_paths = wgsmetadata[wgsmetadata['isabl_sample_id'] == isabl_tumor_id]
    assert wgsmeta_paths.shape[0] == 1
    
    wgsmeta_path = wgsmeta_paths['result_filepath'].values[0]
    inputs_path = os.path.join(wgsmeta_path.rsplit('/', 2)[0], 'inputs.yaml') 
    assert os.path.exists(inputs_path)
    inputs = yaml.load(open(inputs_path, 'r').read(), Loader=yaml.Loader)
    assert isabl_tumor_id in inputs
    isabl_normal_id = inputs[isabl_tumor_id]['normal_id']
    return isabl_normal_id

def get_isabl_snv(consensus_path):
    consensus = pd.read_csv(consensus_path, dtype={'Chromosome': str}, sep='\t', comment='#', low_memory=False)
    consensus = consensus[consensus['Variant_Type'] == 'SNP']
    consensus = consensus
    return consensus

def get_consensus_snv(row, wgssnv_metadata, isabl_maf_cols=isabl_maf_cols, filter_isabl_maf=False):
    isabl_tumor_id = row['isabl_sample_id']
    isabl_normal_id = extract_isabl_normal_id(wgssnv_metadata, isabl_tumor_id)
    
    consensus_path = row['result_filepath']
    path_tumor_id_pat = re.search('([^\/]+)_consensus_somatic.maf', consensus_path)
    assert path_tumor_id_pat
    path_tumor_id = path_tumor_id_pat.groups()[0]
    assert isabl_tumor_id == path_tumor_id
    isabl_normal_id = extract_isabl_normal_id(wgssnv_metadata, isabl_tumor_id)
    
    consensus = pd.read_csv(consensus_path, dtype={'Chromosome': str}, sep='\t', comment='#', low_memory=False)
    consensus = consensus[consensus['Variant_Type'] == 'SNP']
    if filter_isabl_maf == 'AF-filter':
        consensus = consensus[consensus['AF'].fillna(0) < 0.01] # AF-filter
    elif filter_isabl_maf == 'rsID-filter':
        consensus = consensus[consensus['dbSNP_RS'] == 'novel'] # rsID-filter
    elif filter_isabl_maf == 'tempo-filter':
        consensus = consensus[ (consensus['t_alt_count'] / consensus['t_depth']) > 0.05 ] # t_var_freq > .05
        # .05 > t_var_freq > .02 for whitelisted hotspots, which are identified with taylorlab/annotateMaf package
        consensus = consensus[consensus['t_depth'] > 20] # tumor_depth > 20
        consensus = consensus[consensus['t_alt_count'] > 3] # Tumor_count > 3
        consensus = consensus[consensus['n_depth'] > 10] # Normal_depth > 10
        consensus = consensus[consensus['n_ref_count'] > 3] # Normal_count > 3
        consensus = consensus[consensus['AF'].fillna(0) < 0.01] # AF-filter # Gnomad_allele_frequency[‘non_cancer_AF_popmax’]  < .01
        # EncodeDacMapability and RepeatMasker flags
        # MQ > 55 for non-SNP variants
        # Strand bias
        # Multiallelic flags set by Strelka and Mutect2. Strelka does not natively add such a flag, so we calculate it separately.
        
    consensus = consensus[isabl_maf_cols].drop_duplicates()
    
    return isabl_tumor_id, isabl_normal_id, consensus

def get_snv_data_and_sets(joint_wgssnv, wgssnv_metadata, tempo_snv_data,
                          isabl_inspect_cols=dst_maf_cols,
                          plot_venn=False, print_header=True, select_id=None, get_subsets=False,
                          filter_isabl_maf=False):
    header = '\t'.join(['isabl_tumor_id', 'tempo_tumor_id', 'isabl_normal_id', 'tempo_normal_id', 'A', 'B', 'A-B', 'B-A', 'A&B', 'A|B'])
    if print_header: print(header)
    for rix, row in joint_wgssnv.iterrows(): # Only for the overlap between Tempo and Isabl/WGS
        if select_id != None:
            if row['isabl_sample_id'] != select_id: continue #'SA1047A': continue ##@##
        isabl_tumor_id, isabl_normal_id, consensus = get_consensus_snv(row, wgssnv_metadata, filter_isabl_maf=filter_isabl_maf)
        consensus = consensus[isabl_maf_cols].rename(columns=dict(zip(isabl_maf_cols, dst_maf_cols)), inplace=False)
        tempo, tempo_tumor_id, tempo_normal_id = map_tempo_snv_by_id(tempo_snv_data, isabl_tumor_id, isabl_normal_id)

        snv_match = pd.merge(consensus, tempo, how='inner', on=['chrom', 'pos', 'ref', 'alt']).set_index(['chrom', 'pos', 'ref', 'alt'])

        consensus = consensus.set_index(['chrom', 'pos', 'ref', 'alt'], drop=False)
        tempo = tempo.set_index(['chrom', 'pos', 'ref', 'alt'], drop=False)
        consensus_match = consensus[consensus.index.isin(snv_match.index)]
        consensus_nonmatch = consensus[~consensus.index.isin(snv_match.index)]
        tempo_match = tempo[tempo.index.isin(snv_match.index)]
        tempo_nonmatch = tempo[~tempo.index.isin(snv_match.index)]

        A, B, field = print_set_stats(consensus_match, consensus_nonmatch, tempo_match, tempo_nonmatch,
                               isabl_tumor_id, tempo_tumor_id, isabl_normal_id, tempo_normal_id,
                               vartype='snv')
        if plot_venn:
                vd = venn2_unweighted([A, B], set_labels=('consensus', 'tempo'))
        
    if get_subsets:
        return (consensus, consensus_match, consensus_nonmatch, 
                tempo, tempo_match, tempo_nonmatch)
    else:
        return consensus, tempo, field #

def get_tempo_snv_data():
    data_dir = '/juno/work/tempo/wgs/SV/Results' 
    snv_tool = 'combined_mutations'
    somatic_dirs = glob.glob(f"{data_dir}/somatic/*__*")
    tumor_normal_ids = []
    tumor_ids = []
    tempo_snv_data = {}
    for somatic_dir in somatic_dirs:
        tumor_normal = os.path.split(somatic_dir)[-1]
        snv_tool_path = os.path.join(somatic_dir, snv_tool)
        assert os.path.isdir(snv_tool_path)
        snv_path = glob.glob(os.path.join(snv_tool_path, '*.somatic.final.maf'))
        if len(snv_path) != 1: continue
        snv_path = snv_path[0]
        snv_filename = os.path.split(snv_path)[-1]
        snv_tumor_normal = snv_filename.split('.')[0]
        assert tumor_normal == snv_tumor_normal
        tumor_id, normal_id = tumor_normal.split('__')
        tumor_normal_ids.append((tumor_id, normal_id))
        tumor_ids.append(tumor_id)
        # Proc SNV file
        tempo_snv_data[tumor_id] = (normal_id, snv_path)
    return tempo_snv_data, tumor_ids

def get_wgs_metadata():
    wgssnv_path = '/home/chois7/tables/all.WGS-SOMATICCALLING_paths.tsv'
    wgssnv = pd.read_table(wgssnv_path, low_memory=False)
    wgssnv_metadata = wgssnv[wgssnv['result_type'] == 'metadata']
    wgssnv = wgssnv[wgssnv['result_type'] == 'consensus_somatic_maf']
    return wgssnv, wgssnv_metadata

if __name__ == "__main__":
    tempo_snv_data, tumor_ids = get_tempo_snv_data()
    wgssnv, wgssnv_metadata = get_wgs_metadata()
    joint_wgssnv = wgssnv[wgssnv['isabl_sample_id'].isin(tumor_ids)]

    flt = 'tempo-filter'
    save_dir = '/juno/work/shah/users/chois7/tickets/tempo/signature_plot_snv'
    profile_path = f'{save_dir}/profile_SNV.{flt}.tsv'
    profile_data = {}
    field_path = f'{save_dir}/field_SNV.{flt}.tsv'
    fields = {}
    field_ix = 'isabl_tumor_id isabl_normal_id tempo_tumor_id tempo_normal_id A B A-B B-A A&B A|B'.split(' ')
    for ix, isabl_tumor_id in enumerate(joint_wgssnv['isabl_sample_id']):
        print(f'Running for {isabl_tumor_id}', file=sys.stderr)
        isabl, tempo, field = get_snv_data_and_sets(joint_wgssnv, wgssnv_metadata, tempo_snv_data, 
                select_id=isabl_tumor_id, plot_venn=False, print_header=False,
                filter_isabl_maf=flt)
        fields[isabl_tumor_id] = pd.Series(field, index=field_ix)
        isabl_counts, isabl_context = count_snvs(isabl, genome)
        tempo_counts, tempo_context = count_snvs(tempo, genome)
        save_path = os.path.join(save_dir, f'{isabl_tumor_id}.jpg')
        plot_paired_snv_terms(isabl_tumor_id, 
                              pd.DataFrame(isabl_counts, columns=['probability']),
                              pd.DataFrame(tempo_counts, columns=['probability']),
                              save_path=save_path, tags=['isabl', 'tempo'])
        profile_data[f'shahlab_{isabl_tumor_id}'] = isabl_counts
        profile_data[f'tempo_{isabl_tumor_id}'] = tempo_counts
        if ix == 2:
            break

    field_df = pd.DataFrame(fields)
    field_df.T.to_csv(field_path, sep='\t', index=False)
    profile_df = pd.DataFrame(profile_data)
    profile_df.to_csv(profile_path, sep='\t', index_label='SV_type')
