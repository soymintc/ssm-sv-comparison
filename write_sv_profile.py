import os
import re
import sys
import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib
from matplotlib_venn import venn2, venn2_unweighted
import matplotlib.pyplot as plt
import wgs_analysis.algorithms.rearrangement

sv_cols = ["CHROM_A", "STRAND_A", "START_A", "CHROM_B", "STRAND_B", "START_B", 'ID', 'TYPE']
isabl_cols = ['chromosome_1', 'strand_1', 'position_1', 'chromosome_2', 'strand_2', 'position_2', 'prediction_id', 'type']
sv_inspect_cols = isabl_cols + ['INFO_A']
isabl_inspect_cols = isabl_cols + ['num_unique_reads', 'num_split', 'break_distance']

isabl_maf_cols = ['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2',]
dst_maf_cols = ['chrom', 'pos', 'ref', 'alt']

tempo_sv_header = ["CHROM_A", "START_A", "END_A", "CHROM_B", "START_B", "END_B", 
                   "ID", "QUAL", "STRAND_A", "STRAND_B", "TYPE", "FILTER", "NAME_A", "REF_A", "ALT_A", "NAME_B", "REF_B", "ALT_B", "INFO_A", "INFO_B", 
                   "FORMAT", "TUMOR", "NORMAL", "POTENTIAL_CDNA_CONTAMINATION", 
                   "gene1", "transcript1", "site1", "gene2", "transcript2", "site2", "fusion", "Cosmic_Fusion_Counts", 
                   "repName-repClass-repFamily:-site1", "repName-repClass-repFamily:-site2", 
                   "CC_Chr_Band", "CC_Tumour_Types(Somatic)", "CC_Cancer_Syndrome", "CC_Mutation_Type", "CC_Translocation_Partner", 
                   "DGv_Name-DGv_VarType-site1", "DGv_Name-DGv_VarType-site2",]

svlen_bins = [-10, 1e4, 1e5, 1e6, 1e7, np.inf]
svlen_bin_labels = ['<10kb', '10kb-100kb', '100kb-1Mb', '1Mb-10Mb', '>10Mb']

tempo_type_converter = {'DEL': 'del', 'DUP': 'dup', 'INV': 'inv', 'BND': 'tr'}
isabl_type_converter = {'duplication': 'dup', 'translocation': 'tr', 'deletion': 'del', 'inversion': 'inv'}

sv_labels = ['del:<10kb', 'del:10kb-100kb', 'del:100kb-1Mb', 'del:1Mb-10Mb', 'del:>10Mb', 
             'dup:<10kb', 'dup:10kb-100kb', 'dup:100kb-1Mb', 'dup:1Mb-10Mb', 'dup:>10Mb', 
             'inv:<10kb', 'inv:10kb-100kb', 'inv:100kb-1Mb', 'inv:1Mb-10Mb', 'inv:>10Mb', 'tr']

sv_colors = ['#d6e6f4', '#abd0e6', '#6aaed6', '#3787c0', '#105ba4', 
             '#fedfc0', '#fdb97d', '#fd8c3b', '#e95e0d', '#b63c02', 
             '#dbf1d6', '#aedea7', '#73c476', '#37a055', '#0b7734', '#9467BD']

def map_tempo_sv_by_id(isabl_tumor_id, isabl_normal_id, sv_inspect_cols=sv_inspect_cols,
                      tempo_type_converter=tempo_type_converter, 
                      svlen_bins=svlen_bins, svlen_bin_labels=svlen_bin_labels):
    assert isabl_tumor_id in tempo_data
    tempo_normal_id, tempo_sv_path = tempo_data[isabl_tumor_id]
    tempo_tumor_id = isabl_tumor_id
    if isabl_normal_id != tempo_normal_id:
        print(f"WARNING: isabl ({isabl_normal_id}) != tempo ({tempo_normal_id})", file=sys.stderr)
    tempo = get_tempo_sv(tempo_sv_path)
    if 'FORMAT' not in tempo.columns:
        print(isabl_tumor_id, tempo)
    tempo = proc_tempo_sv(tempo)
    
    tempo['sv_type'] = tempo['type'].replace(tempo_type_converter)
    tempo['sv_length'] = pd.cut(tempo['SVLEN'], bins=svlen_bins, labels=svlen_bin_labels).values
    tempo['sv_category'] = tempo['sv_type'].str.cat(tempo['sv_length'], sep=':')
    return tempo, tempo_tumor_id, tempo_normal_id

def make_sv_set(data, vartype='snv'):
    if vartype == 'snv':
        ix_cols = ['chrom', 'pos', 'ref', 'alt']
    elif vartype == 'sv':
        ix_cols = ['chromosome_1', 'position_1', 'strand_1', 'chromosome_2', 'position_2', 'strand_2']
    df_ix = data.copy().set_index(ix_cols).index.tolist()
    df_ix_set = set(df_ix)
    # assert len(df_ix_set) == len(df_ix)
    return df_ix_set

def print_set_stats(consensus_match, consensus_nonmatch, tempo_match, tempo_nonmatch,
                  isabl_tumor_id, tempo_tumor_id, isabl_normal_id, tempo_normal_id, vartype='snv'):
    A_and_B = make_sv_set(consensus_match, vartype=vartype)
    A_and_B_from_tempo = make_sv_set(tempo_match, vartype=vartype)
    assert abs(len(A_and_B) - len(A_and_B_from_tempo)) <= 10, f'A_and_B:\n{A_and_B}\nA_and_B_from_tempo:\n{A_and_B_from_tempo}\n'
    A_not_B = make_sv_set(consensus_nonmatch, vartype=vartype)
    B_not_A = make_sv_set(tempo_nonmatch, vartype=vartype)
    
    A = A_and_B.union(A_not_B)
    B = A_and_B.union(B_not_A)
    
    field = [isabl_tumor_id, tempo_tumor_id, isabl_normal_id, tempo_normal_id, 
             len(A), len(B), len(A-B), len(B-A), len(A&B), len(A|B)]
    field = [str(_) for _ in field]
    return A, B, field # for venn2 plott

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

def get_consensus_sv(row, wgsmetadata, isabl_cols=isabl_cols, 
                     isabl_type_converter=isabl_type_converter,
                     svlen_bins=svlen_bins, svlen_bin_labels=svlen_bin_labels):
    """Input: metadata row"""
    isabl_tumor_id = row['isabl_sample_id']
    consensus_path = row['result_filepath']
    path_tumor_id_pat = re.search('([^\/]+)_filtered_consensus_calls.csv.gz', consensus_path)
    assert path_tumor_id_pat
    path_tumor_id = path_tumor_id_pat.groups()[0]
    assert isabl_tumor_id == path_tumor_id
    isabl_normal_id = extract_isabl_normal_id(wgsmetadata, isabl_tumor_id)
    
    consensus = pd.read_csv(consensus_path, dtype={'chromosome_1': str, 'chromosome_2': str, 'prediction_id':str}, low_memory=False)
    
    return isabl_tumor_id, isabl_normal_id, consensus

def get_tempo_sv(sv_path, header=tempo_sv_header, sv_cols=sv_cols, isabl_cols=isabl_cols):
    sv = pd.read_table(sv_path, comment='#', names=header, dtype={'CHROM_A': str, 'CHROM_B': str}, low_memory=False)
    sv.rename(columns=dict(zip(sv_cols, isabl_cols)), inplace=True)
    assert 'TUMOR' in sv.columns
    assert 'NORMAL' in sv.columns
    return sv

def extract_infos(row):
    int_keys = ['NumCallers', 'NumCallersPass', 'SVLEN', 
                'delly_PE', 'delly_SR',]
    info_a = row['INFO_A']
    field = info_a.split(';') 
    infos = {}
    for info in field:
        if '=' in info:
            key, value = info.split('=')
            if key in int_keys:
                value = int(value)
            infos[key] = value
            continue
        else:
            infos[info] = True
            
    formats = row['FORMAT'].split(':')
    t_genotypes = row['TUMOR'].split(':')
    t_genotypes = dict(zip(formats, t_genotypes))
    n_genotypes = row['NORMAL'].split(':')
    n_genotypes = dict(zip(formats, n_genotypes))
    
    infos['t_var_paired_reads'] = 0
    infos['n_var_paired_reads'] = 0
    infos['t_var_split_reads'] = 0
    infos['n_var_split_reads'] = 0
    if 'delly_RR' in formats and 'delly_RV' in formats:
        infos['t_var_paired_reads'] = int(t_genotypes['delly_DV'])
        infos['n_var_paired_reads'] = int(n_genotypes['delly_DV'])
        infos['t_var_split_reads'] = int(t_genotypes['delly_RV'])
        infos['n_var_split_reads'] = int(n_genotypes['delly_RV'])
    elif 'manta_PR' in formats and 'manta_SR' in formats:
        _, infos['t_var_paired_reads'] = [int(x) for x in t_genotypes['manta_PR'].split(',')]
        _, infos['n_var_paired_reads'] = [int(x) for x in n_genotypes['manta_PR'].split(',')]
        _, infos['t_var_split_reads'] = [int(x) for x in t_genotypes['manta_SR'].split(',')]
        _, infos['n_var_split_reads'] = [int(x) for x in n_genotypes['manta_SR'].split(',')]
    else:
        pass
        
    return infos

def proc_tempo_sv(df): 
    num_callers = []
    num_callers_pass = []
    split_read_supports = []
    sv_len = []
    
    t_var_paired_reads, n_var_paired_reads, t_var_split_reads, n_var_split_reads = [], [], [], []
    
    assert 'FORMAT' in df.columns, f'{tumor_id}\n{df}'
    for rix, row in df.iterrows():
        infos = extract_infos(row)
        # print(infos)
        t_var_paired_reads.append(infos['t_var_paired_reads'])
        n_var_paired_reads.append(infos['n_var_paired_reads'])
        t_var_split_reads.append(infos['t_var_split_reads'])
        n_var_split_reads.append(infos['n_var_split_reads'])

        # return infos
        # return infos, n_genotypes, t_genotypes ##@##
        num_callers.append(infos['NumCallers'])
        num_callers_pass.append(infos['NumCallersPass'])
        sv_len_val = np.inf
        if 'SVLEN' in infos:
            sv_len_val = infos['SVLEN']
        sv_len.append(sv_len_val)
    
    df['NumCallers'] = num_callers
    df['NumCallersPass'] = num_callers_pass
    df['SVLEN'] = sv_len
    df['t_var_paired_reads'] = t_var_paired_reads
    df['n_var_paired_reads'] = n_var_paired_reads
    df['t_var_split_reads'] = t_var_split_reads
    df['n_var_split_reads'] = n_var_split_reads
    
    return df

def draw_sv_spectra(svdf, sample_id, tag, 
                    sv_labels=sv_labels, sv_colors=sv_colors):
    """Draw SV spectra plot based on tempo or consensus SV dataframe"""
    
    df = svdf.copy() #
    df.loc[df['sv_type'] == 'tr', 'sv_category'] = 'tr'

    fig, ax = plt.subplots(1)
    fig.set_figheight(2.5)
    fig.set_figwidth(4)

    title = sample_id
    if tag: title += ' ' + tag
    fig.suptitle(title)

    sv_counts = pd.Series([0] * len(sv_labels), index=sv_labels)
    sv_count_values = sv_counts.index.map(df['sv_category'].value_counts()).fillna(0)
    sv_counts = pd.Series(sv_count_values, index=sv_labels).astype(int)
    # print(sv_counts)
    plot_x_index = range(len(sv_counts))

    font = matplotlib.font_manager.FontProperties()
    font.set_family('monospace')

    ax.bar(height=sv_counts, x=range(sv_counts.shape[0]), color=sv_colors)
    plt.xticks(plot_x_index, sv_counts.index, rotation=90, fontproperties=font)
    plt.xlim((-1, len(sv_counts)))
    
def get_sv_category_counts(df, sv_labels=sv_labels):
    assert 'sv_category' in df
    sv_counts = pd.Series([0] * len(sv_labels), index=sv_labels)
    sv_counts = pd.Series([0] * len(sv_labels), index=sv_labels)
    sv_count_values = sv_counts.index.map(df['sv_category'].value_counts()).fillna(0)
    sv_counts = pd.Series(sv_count_values, index=sv_labels).astype(int)
    plot_x_index = range(len(sv_counts))
    font = matplotlib.font_manager.FontProperties()
    font.set_family('monospace')
    assert df.shape[0] == sv_counts.sum(), f'df={df}\nsv_counts={sv_counts}'
    return sv_counts, plot_x_index, font
    
def draw_paired_sv_spectra(svdf1, svdf2, sample_id, tag, 
                    sv_labels=sv_labels, sv_colors=sv_colors, save_path=False):
    """Draw SV spectra plots based on tempo or consensus SV dataframe"""
    
    df1 = svdf1.copy() #
    df1.loc[df1['sv_type'] == 'tr', 'sv_category'] = 'tr'
    df2 = svdf2.copy() #
    df2.loc[df2['sv_type'] == 'tr', 'sv_category'] = 'tr'

    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(4)
    fig.set_figwidth(9)

    title = sample_id
    if tag: 
        assert tag.count(' vs ')
        title1, title2 = tag.replace('(','').replace(')','').split(' vs ')
    fig.suptitle(title)

    sv_counts1, plot_x_index, font = get_sv_category_counts(df1)
    counts_max1 = sv_counts1.max()
    ax[0].bar(height=sv_counts1, x=range(sv_counts1.shape[0]), color=sv_colors)
    ax[0].set_xticks(np.arange(len(sv_counts1.index))) # show all labels
    ax[0].set_xticklabels(sv_counts1.index, rotation=90, fontproperties=font)
    ax[0].set_title(title1)
    
    sv_counts2, plot_x_index, font = get_sv_category_counts(df2)
    counts_max2 = sv_counts2.max()
    ax[1].bar(height=sv_counts2, x=range(sv_counts2.shape[0]), color=sv_colors)
    ax[1].set_xticks(np.arange(len(sv_counts2.index))) # show all labels
    ax[1].set_xticklabels(sv_counts2.index, rotation=90, fontproperties=font)
    ax[1].set_title(title2)
    
    plt.xlim((-1, len(sv_counts2)))
    ax[0].set_ylim((0, 2+max(counts_max1, counts_max2)))
    ax[1].set_ylim((0, 2+max(counts_max1, counts_max2)))
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    
    return sv_counts1, sv_counts2

def get_tempo_sv_paths():
    sv_tools = ['facets', 'brass', 'manta', 'delly', 'combined_svs', 'svaba']
    sv_tool = 'combined_svs'
    sv_dirs = glob.glob("/juno/work/tempo/wgs/SV/Results/somatic/*__*")
    tumor_normal_ids = {}
    tumor_sv_ids = []
    tempo_data = {}
    for sv_dir in sv_dirs:
        tumor_normal = os.path.split(sv_dir)[-1]
        sv_tool_path = os.path.join(sv_dir, sv_tool)
        assert os.path.isdir(sv_tool_path)
        sv_path = glob.glob(os.path.join(sv_tool_path, '*combined.annot.filtered.pass.bedpe'))
        assert len(sv_path) == 1
        sv_path = sv_path[0]
        sv_filename = os.path.split(sv_path)[-1]
        sv_tumor_normal = sv_filename.split('.')[0]
        assert tumor_normal == sv_tumor_normal
        tumor_id, normal_id = tumor_normal.split('__')
        tumor_normal_ids[tumor_id] = (tumor_id, normal_id)
        tumor_sv_ids.append(tumor_id)
        
        # Proc SV file
        tempo_data[tumor_id] = (normal_id, sv_path)
    return tumor_sv_ids, tempo_data

def apply_mmctm_sv_filters(sv):
    sv = sv.copy()
    sv = sv[
        ~sv['is_germline'] &
        ~sv['is_filtered'] &
        ~sv['is_low_mappability'] &
        pd.isnull(sv['dgv_ids']) &
        (sv['template_length_min'] >= 200) &
        (sv['num_unique_reads'] >= 5) &
        (sv['num_split'] >= 2)
    ]
    
    return sv

def apply_tempo_dgv_filter(tempo):
    df = tempo.copy()
    df['filter_out'] = False
    
    tempo_to_dgv_converter = {
        'DEL': 'Deletion',
        'DUP': 'Duplication',
        'BND': 'Complex',
        'INV': 'Inversion',
    }
    for rix, row in df.iterrows():
        filter_out_row = False
        
        dgv1 = row['DGv_Name-DGv_VarType-site1']
        dgv2 = row['DGv_Name-DGv_VarType-site2']
    
        dset1 = set(dgv1.split('<=>'))
        dset2 = set(dgv2.split('<=>'))
        dboth = dset1 & dset2
        
        if dboth:
            dgv_types = list(set([x.split('-')[-1] for x in dboth]))
            if tempo_to_dgv_converter[row['type']] in dgv_types:
                df.loc[rix, 'filter_out'] = True
    
    return df.loc[df['filter_out'] != True, :]

def get_wgssv_results():
    wgssv_path = '/home/chois7/tables/all.WGS-BREAKPOINTCALLING_paths.tsv'

    wgssv = pd.read_table(wgssv_path, low_memory=False)
    wgsmetadata = wgssv[wgssv['result_type'] == 'metadata']
    wgssv = wgssv[wgssv['result_type'] == 'consensus_calls']
    return wgssv, wgsmetadata

def get_sv_data_and_sets(joint_wgssv, wgsmetadata, 
                         isabl_inspect_cols=isabl_inspect_cols,
                         isabl_type_converter=isabl_type_converter,
                         svlen_bin=svlen_bins, svlen_bin_labels=svlen_bin_labels,
                         plot_venn=False, print_header=True, select_id=None, get_subsets=False,
                         filter_svs=False):
    header = '\t'.join(['isabl_tumor_id', 'tempo_tumor_id', 'isabl_normal_id', 'tempo_normal_id', 'A', 'B', 'A-B', 'B-A', 'A&B', 'A|B'])
    if print_header: print(header)
    for rix, row in joint_wgssv.iterrows(): # Only for the overlap between Tempo and Isabl/WGS
        if select_id != None:
            if row['isabl_sample_id'] != select_id: continue #'SA1047A': continue ##@##

        isabl_tumor_id, isabl_normal_id, consensus = get_consensus_sv(row, wgsmetadata)
        tempo, tempo_tumor_id, tempo_normal_id = map_tempo_sv_by_id(isabl_tumor_id, isabl_normal_id)
        
        if filter_svs:
            consensus = apply_mmctm_sv_filters(consensus)
            tempo = tempo[(tempo['n_var_paired_reads'] == 0) & (tempo['n_var_split_reads'] == 0)] # is_germline filter
            tempo = tempo[(tempo['t_var_paired_reads'] >= 5) & (tempo['t_var_split_reads'] >= 2)] # tumor paired and split reads filter
            # tempo = tempo[(tempo['DGv_Name-DGv_VarType-site1'] == '.') | (tempo['DGv_Name-DGv_VarType-site2'] == '.')] # DGV filter
            tempo = apply_tempo_dgv_filter(tempo)

        consensus = consensus[isabl_inspect_cols]
        consensus['sv_type'] = consensus['type'].replace(isabl_type_converter)
        consensus['sv_length'] = pd.cut(consensus['break_distance'], bins=svlen_bins, labels=svlen_bin_labels).values
        consensus['sv_category'] = consensus['sv_type'].str.cat(consensus['sv_length'], sep=':')
        
        try:
            sv_match = wgs_analysis.algorithms.rearrangement.match_breakpoints(consensus, tempo, window_size=200)
        except ValueError:
            sv_match = pd.DataFrame(columns = ['reference_id', 'target_id'])

        consensus_match = consensus[consensus['prediction_id'].isin(sv_match['reference_id'])]
        consensus_nonmatch = consensus[~consensus['prediction_id'].isin(sv_match['reference_id'])]
        tempo_match = tempo[tempo['prediction_id'].isin(sv_match['target_id'])]
        tempo_nonmatch = tempo[~tempo['prediction_id'].isin(sv_match['target_id'])]
        A, B, field = print_set_stats(consensus_match, consensus_nonmatch, tempo_match, tempo_nonmatch,
                               isabl_tumor_id, tempo_tumor_id, isabl_normal_id, tempo_normal_id,
                               vartype='sv')
        if plot_venn:
            vd = venn2_unweighted([A, B], set_labels=('consensus', 'tempo'))
        
    if get_subsets:
        return (consensus, consensus_match, consensus_nonmatch, 
                tempo, tempo_match, tempo_nonmatch)
    else:
        return consensus, tempo, field #


if __name__ == '__main__':
    wgssv, wgsmetadata = get_wgssv_results()
    tumor_sv_ids, tempo_data = get_tempo_sv_paths()
    joint_wgssv = wgssv[wgssv['isabl_sample_id'].isin(tumor_sv_ids)]

    save_dir = '/juno/work/shah/users/chois7/tickets/tempo/signature_plot'
    profile_path = f'{save_dir}/profile_SV.tsv'
    profile_data = {}
    field_path = f'{save_dir}/field_SV.tsv'
    field_ix = 'isabl_tumor_id isabl_normal_id tempo_tumor_id tempo_normal_id A B A-B B-A A&B A|B'.split(' ')
    fields = {} 
    for isabl_tumor_id in joint_wgssv['isabl_sample_id']: # []:
        print(f'Running for {isabl_tumor_id}', file=sys.stderr)
        consensus, tempo, field = get_sv_data_and_sets(joint_wgssv, wgsmetadata, select_id=isabl_tumor_id, plot_venn=False, print_header=False, filter_svs=True)
        field = pd.Series(field, index=field_ix)
        fields[isabl_tumor_id] = field
        save_path = os.path.join(save_dir, f'{isabl_tumor_id}.jpg')
        shahlab_profile, tempo_profile = draw_paired_sv_spectra(consensus, tempo, isabl_tumor_id, '(shahlab vs tempo)', save_path=save_path)
        profile_data[f'shahlab_{isabl_tumor_id}'] = shahlab_profile
        profile_data[f'tempo_{isabl_tumor_id}'] = tempo_profile
        
    field_df = pd.DataFrame(fields)
    field_df.T.to_csv(field_path, sep='\t', index=False)
    profile_df = pd.DataFrame(profile_data)
    profile_df.to_csv(profile_path, sep='\t', index_label='SV_type')
