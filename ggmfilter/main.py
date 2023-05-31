import pathlib
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import pysam
import time

'''
ToDo
- Start, End,列作る

'''


def parser_setting():
    parser = argparse.ArgumentParser(description = 'GGM filtering tool')
    parser.add_argument('--input', '-i', required=True, 
                        type=pathlib.Path, help='path to exome_summary file')
    parser.add_argument('--resources', '-r', required=True, 
                        type=pathlib.Path, help='path to resources directory')
    parser.add_argument('--gq', '-q', default=20, 
                        type=pathlib.Path, help='GQ')
    parser.add_argument('--dp', '-d', default=10, 
                        type=pathlib.Path, help='DP')
    parser.add_argument('--ad', '-a', default=0.3, 
                        type=pathlib.Path, help='AD')
    args = vars(parser.parse_args())

    return args


def load_file(input_file: str, type: str) -> pd.DataFrame:
    if type == 'csv':
        df = pd.read_csv(input_file, header=0, dtype=str)
    elif type == 'tsv':
        df = pd.read_csv(input_file, header=0, dtype=str, sep='\t')

    return df


def print_filtering_count(func):
    def _wrapper(*args, **kwargs):
        print(f'Execute   : {func.__name__}')
        ts = time.perf_counter()
        pre = len(args[0])
        result = func(*args, **kwargs)
        te = time.perf_counter()
        post = len(result)
        print(f'Filtering : {pre} -> {post}')
        print(f'Run-time  : {te - ts}\n')
        return result
    return _wrapper


def _remove_ver(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].apply(lambda x: x.split('.')[0] if '.' in x else x)
    return df


def pre_proceesing(df: pd.DataFrame) -> pd.DataFrame:
    for rel in ['pro', 'pat', 'mat']:
        df = pd.concat(
            [df, df[f'GQ:DP:AD({rel})'].str.split(':', expand=True)], axis=1)
        for i in range(3):
            df[i] = df[i].replace('.', np.nan)
            df[i] = df[i].replace('-', np.nan)
        df = df.astype({0: float, 1: float, 2: float})
        df.rename(columns={0: f'GQ({rel})', 1: f'DP({rel})', 2: f'AD({rel})'}, 
                  inplace=True)
    df = pd.concat(
        [df, df[f'GGM(AC/AN)'].str.split('/', expand=True)], axis=1)
    df = df.astype({0: float, 1: float})
    df.rename(columns={0: 'GGM.AC', 1: 'GGM.AN'}, inplace=True)

    df['variant_id'] = df['Chr'] + '-' + df['Position'] + '-' \
                     + df['Ref'] + '-' + df['Alt']

    df['GGM.AF'] = np.nan
    df['SpliceAI_INFO'] = np.nan
    df['max_splai'] = np.nan
    df['REVEL'] = np.nan

    return df

@print_filtering_count
def quality_check(df: pd.DataFrame, gq: int, dp: int) -> pd.DataFrame:
    df = df[(df['GQ(pro)'] >= gq) 
            & (df['DP(pro)'] >= dp)
            & (df['GQ(pat)'] >= gq)
            & (df['DP(pat)'] >= dp)
            & (df['GQ(mat)'] >= gq)
            & (df['DP(mat)'] >= dp)]
    return df

@print_filtering_count
def exclude_low_ad(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df['prc.AD'] = df['AD(pro)'].div(df['DP(pro)'], axis=0)
    df = df[df['prc.AD'] > threshold]
    return df

@print_filtering_count
def exclude_identified_variants(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Analysis status'] != 'Identified']
    return df

@print_filtering_count
def exclude_ggm_common(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df['GGM.AF'] = df['GGM.AC'].div(df['GGM.AN'], axis=0)
    df = df[df['GGM.AF'] < threshold]

    return df


@print_filtering_count
def exclude_low_splai(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.astype({'max_splai': float})
    df = df[~(((df['Effect'] == 'intron_variant')
               | (df['Effect'] == 'synonymous_variant')
               | (df['Effect'] == '5_prime_UTR_variant')
               | (df['Effect'] == '3_prime_UTR_variant')
               | (df['Effect'] == 'upstream_gene_variant')
               | (df['Effect'] == 'downstream_gene_variant'))
               & (df['max_splai'] <= threshold))]
    
    return df

def annotate_gnomad(df: pd.DataFrame) -> pd.DataFrame:
    gnomad = load_file(input_file=gnomad_file, type='tsv')
    gnomad = gnomad[['gene', 'transcript', 'pLI', 'pRec', 'syn_z', 'mis_z', 'oe_lof_upper']]
    gnomad = gnomad.rename(columns={'gene': 'gnomAD_gene', 'oe_lof_upper': 'LOEUF', 
                                    'transcript': 'Ense.canonical'})
    df = pd.merge(df, gnomad, left_on='Gene', right_on='gnomAD_gene', how='left')

    return df


def annotate_hgmd(df: pd.DataFrame) -> pd.DataFrame:
    hgmd = load_file(input_file=hgmd_file, type='tsv')
    hgmd = hgmd[['GeneSymbol', 'expected_inheritance', 'DM', 'disease', 'omimid']]
    hgmd = hgmd.rename(columns={'expected_inheritance': 'exp.MOI', 
                                'DM': 'DM', 'disease': 'HGMD.disease'})
    df = pd.merge(df, hgmd, left_on='Gene', right_on='GeneSymbol', how='left')

    return df


def annotate_mane(df: pd.DataFrame) -> pd.DataFrame:
    mane = load_file(input_file=mane_file, type='tsv')
    mane = mane[['Ensembl_nuc', 'Ensembl_prot', 'symbol']]
    mane = mane.rename(columns={'Ensembl_nuc': 'MANE_ENST', 'Ensembl_prot': 'ENSP'})
    df = pd.merge(df, mane, left_on='Gene', right_on='symbol', how='left')
    df = _remove_ver(df, 'MANE_ENST')
    df = _remove_ver(df, 'ENSP')

    return df


def annotate_splai(df: pd.DataFrame) -> pd.DataFrame:
    print('Annotating SpliceAI scores')
    splai_snv = '/bulk/SpliceAI/spliceai_scores.masked.snv.hg19.vcf.gz'
    splai_indel = '/bulk/SpliceAI/spliceai_scores.masked.indel.hg19.vcf.gz'
    tbx_snv = pysam.TabixFile(splai_snv)
    tbx_indel = pysam.TabixFile(splai_indel)
    chr_col = df.columns.get_loc('Chr')
    pos_col = df.columns.get_loc('Position')
    splai_col = df.columns.get_loc('SpliceAI_INFO')
    var_id_sr = df['variant_id']  

    for index, variant_id in enumerate(tqdm(var_id_sr)): 
        str_contig = df.iat[index, chr_col]

        if str_contig == 'X':
            query_contig = str_contig
            pass
        else:
            query_contig = int(str_contig)
            pass

        query_pos_start = int(df.iat[index, pos_col]) - 1
        query_pos_end = query_pos_start + 2

        for row in tbx_snv.fetch(query_contig, query_pos_start, 
                                 query_pos_end, parser=pysam.asVCF()):
            pos_snv = row.pos + 1
            splai_id = f'{row.contig}-{pos_snv}-{row.ref}-{row.alt}'
            if variant_id == splai_id:
                df.iat[index, splai_col] = row.info.replace('SpliceAI=', '')
                break
            else:
                pass
        
        if df.iat[index, splai_col]:
            pass
        else:
            for row in tbx_indel.fetch(query_contig, query_pos_start, 
                                       query_pos_end, parser=pysam.asVCF()):
                pos_indel = row.pos + 1
                splai_id = f'{row.contig}-{pos_indel}-{row.ref}-{row.alt}'
                if variant_id == splai_id :
                    df.iat[index, splai_col] = row.info.replace('SpliceAI=', '')
                    break
                else:
                    pass

    df = pd.concat([df, df['SpliceAI_INFO'].str.split('|', expand=True)], axis=1)
    df.rename(columns={0: 'splai_alt', 1: 'splai_symbol', 2: 'acp.gain', 
                       3: 'acp.loss', 4: 'dnr.gain', 5: 'dnr.loss',
                       6: 'acp.gain_pos', 7: 'acp.loss_pos', 8: 'dnr.gain_pos',
                       9: 'dnr.loss_pos'}, inplace=True)
    df.drop(columns=['SpliceAI_INFO', 'splai_alt', 'splai_symbol'], inplace=True)
    
    return df


def insert_maxsplai(df: pd.DataFrame) -> pd.DataFrame:
    maxsplai_col = df.columns.get_loc('max_splai')
    splai_scores_cols = ['acp.gain', 'acp.loss', 'dnr.gain', 'dnr.loss']
    sr = df[splai_scores_cols].max(axis=1)
    for index, maxvalue in enumerate(sr):
        df.iloc[index, maxsplai_col] = maxvalue

    return df


def annotate_revel(df: pd.DataFrame) -> pd.DataFrame:
    print('Annotating REVEL scores')
    # revel = '/bulk/REVEL/revel.nonheader.vcf.gz'
    tbx_revel = pysam.TabixFile(revel)
    chr_col = df.columns.get_loc('Chr')
    pos_col = df.columns.get_loc('Position')
    ens_cano_col = df.columns.get_loc('Ense.canonical')
    revel_col = df.columns.get_loc('REVEL')
    var_id_sr = df['variant_id']  

    for index, variant_id in enumerate(tqdm(var_id_sr)): 
        # print(variant_id)
        str_contig = df.iat[index, chr_col]
        
        # Cast to integer
        if str_contig == 'X':
            query_contig = str_contig
            pass
        else:
            query_contig = int(str_contig)
            pass
        
        # Set query position (start-end) and ENST-ID
        query_start = int(df.iat[index, pos_col]) - 1
        query_end = query_start + 2
        query_enst = df.iat[index, ens_cano_col]

        for row in tbx_revel.fetch(query_contig, query_start, 
                                   query_end, parser=pysam.asBed()):
            # row[0]: contig, row[1]: pos, row[2]: Ref, 
            # row[3]: Alt, row[4]: score, row[8]: ENST 
            revel_id = f'{row[0]}-{row[1]}-{row[2]}-{row[3]}'

            if variant_id == revel_id:
                revel_esnt_set = set(row[8].split(sep=';'))
                if query_enst in revel_esnt_set:
                    df.iat[index, revel_col] = row[4]
                    break
                else:
                    pass
            else:
                pass
        else:
            pass
            
    return df


def classify_inheritance_model(df: pd.DataFrame) -> tuple:
    df_denovo = df[(df['exp.MOI'] != 'AR')
                   & (df['Vtype'] == 'denovo')]
    
    df_homo = df[(df['exp.MOI'] != 'AD') 
                 & (df['Vtype'] == 'homo')]
    
    df_ch = df[(df['exp.MOI'] != 'AD')
               & ((df['Vtype'] == 'ch_fa')
                   | (df['Vtype'] == 'ch_ma')
                   | (df['Vtype'] == 'denovo'))]

    return df_denovo, df_ch, df_homo


def _create_sample_gene_col(row):
    return row['Sample'] + '-' + row['Gene']

def _create_sample_gene_var_col(row):
    return row['Sample'] + '-' + row['Gene'] + '-' + row['Vtype']

def _chfunc(row):
    if row['sg_count'] == 1:
        return '.'
    elif row['sg_count'] == row['sgv_count']:
        return '.'
    else:
        return 'PASS'
    
def filtering_ch(df: pd.DataFrame) -> pd.DataFrame:
    df['smpl_gene'] = df.apply(_create_sample_gene_col, axis=1)
    df['smpl_gene_var'] = df.apply(_create_sample_gene_var_col, axis=1)
    series = df.groupby('smpl_gene')['smpl_gene'].transform('count').rename('sg_count')
    df = pd.concat([df, series], axis=1)
    series = df.groupby('smpl_gene_var')['smpl_gene_var'].transform('count').rename('sgv_count')
    df = pd.concat([df, series], axis=1)
    df['CH_filter'] = df.apply(_chfunc, axis=1)
    
    return df

def _filter_low_cadd(x, **kwargs):
    if x >= float(kwargs['cadd']):
        return 'PASS'
    else:
        return '--'

def _filter_low_revel(x, **kwargs):
    if x >= float(kwargs['revel']):
        return 'PASS'
    else:
        return '--'

def revelcadd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace({'CADD': {'-': 9999}})
    df = df.astype({'CADD': float, 'REVEL': float})
    df['CADD_Filter'] = df.apply(_filter_low_cadd, cadd=15)
    df['REVEL_Filter'] = df.apply(_filter_low_revel, revel=0.23)

    return df

@print_filtering_count
def exclude_low_cadd(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.replace({'CADD': {'-': np.nan}})
    df = df.astype({'CADD': float})
    df = df[~((df['CADD'] >= 0) & (df['CADD'] < threshold))]
    df = df.replace({'CADD': {np.nan: '.'}}) 
    return df


@print_filtering_count
def exclude_low_revel(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.astype({'REVEL': float})
    df = df[~((df['REVEL'] >= 0) & (df['REVEL'] < threshold))]
    df = df.replace({'REVEL': {np.nan: '.'}}) 
    return df



def order_cols(df: pd.DataFrame) -> pd.DataFrame:
    # df.drop(columns=df.columns[0])
    re_order_cols = ['Sample', 'Disease', 'HGMD.disease', 
       'Gene', 'DM', 'LOEUF', 'exp.MOI', 'Vtype', 'CH_filter', 
       'Effect', 'variant_id', 'max_splai', 'REVEL', 'CADD', 'SIFT', 
       'PolyPhen-2', 'MutationTaster', 'Impact', 
       'Transcript', 'Amino acid change2',  
       'Chr', 'Position', 'Ref', 'Alt', 'Distance', 'alleleID',
       'GGM(AC/AN)', 'ToMMo3.5K(AC)', 'ToMMo3.5K(AN)', 'ToMMo3.5K(AF)',
       'JPNCTL(SC)', 'JPNCTL(SN)', 'ExAC(AC)', 'ExAC(AF)', 'gnomAD(AC)',
       'gnomAD(AN)', 'gnomAD(AF)', 'Family', 
       'ID(pro)','AS(pro)', 'GT(pro)', 'GQ:DP:AD(pro)', 
       'ID(pat)', 'AS(pat)', 'GT(pat)','GQ:DP:AD(pat)', 
       'ID(mat)', 'AS(mat)', 'GT(mat)', 'GQ:DP:AD(mat)',
       'GQ(pro)','DP(pro)', 'AD(pro)', 
       'GQ(pat)','DP(pat)', 'AD(pat)', 
       'GQ(mat)','DP(mat)', 'AD(mat)', 
       'acp.gain', 'acp.loss', 'dnr.gain', 'dnr.loss', 
       'acp.gain_pos', 'acp.loss_pos', 'dnr.gain_pos','dnr.loss_pos', 
       'omimid', 'MANE_ENST', 'ENSP', 'Analysis status']
    df = df.reindex(columns=re_order_cols)

    return df


def configure_output(input_path: str) -> str:
    input_file = pathlib.Path(input_path) 
    output = f'{input_file.parent}/{input_file.stem}'

    return output


def output_tsv(dfs: list, outputs: list):
    for df, output in zip(dfs, outputs):
        df.to_csv(output, sep='\t', index=False)

def _end_col(row):
    end = abs(int(len(row['REF'])) - int(len(row['ALT']))) + int(row['Start'])
    return end

def renamecol(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={'Ref': 'REF', 'Alt': 'ALT', 
                       'Chr': 'CHROM', 'Position': 'POS'}, inplace=True)
    df['Start'] = df['POS']
    df['End'] = df.apply(_end_col, axis=1)  

    return df

if __name__ == "__main__":
    args = parser_setting()
    input_file =args['input']
    resources = args['resources']
    gq = args['gq']
    dp = args['dp']
    ad = args['ad']
    gnomad_file = f'{resources}/gnomAD_constraint_metrics/gnomad.v2.1.1.lof_metrics.by_gene.txt'
    hgmd_file = f'{resources}/HGMD_info/HGMD_2023.1.dmcount.info.decode.txt'
    mane_file = f'{resources}/MANE_select/MANE.GRCh38.v1.1.summary.txt.gz'
    revel = f'{resources}/REVEL/revel.nonheader.vcf.gz'
    splai_snv = f'{resources}/SpliceAI/spliceai_scores.masked.snv.hg19.vcf.gz'
    splai_indel = f'{resources}/SpliceAI/spliceai_scores.masked.indel.hg19.vcf.gz'

    print(f'\nInput file: {input_file}')
    df = load_file(input_file=input_file, type='tsv')
    df = pre_proceesing(df)
    print(f'All variants: {len(df)}\n')

    df = quality_check(df, gq=gq, dp=dp)
    df = exclude_low_ad(df, threshold=ad)
    df = exclude_ggm_common(df, threshold=0.05)
    df = exclude_identified_variants(df)
    df = exclude_low_cadd(df, threshold=15)

    print('Annotating ......\n')
    df = annotate_splai(df)
    df = insert_maxsplai(df)
    df = annotate_gnomad(df)
    df = annotate_hgmd(df)
    df = annotate_mane(df)
    df = annotate_revel(df)

    df = exclude_low_splai(df, threshold=0.1)
    df = exclude_low_revel(df, threshold=0.25)

    df = order_cols(df)
    df = renamecol(df)

    print('Classify by model ......\n')
    df_denovo, df_ch, df_homo = classify_inheritance_model(df)
    df_ch = filtering_ch(df_ch)

    print(f'Remaining Variants: \ndenovo: {len(df_denovo)}\nch    : {len(df_ch)}\nhomo  : {len(df_homo)}\n')
    print(f'\nCompleted!! \n\n')

    output=configure_output(input_file)
    output_list = [f'{output}.denovo.tsv',
                   f'{output}.ch.tsv',
                   f'{output}.homo.tsv']
    output_tsv(dfs=[df_denovo, df_ch, df_homo], 
               outputs=output_list)
