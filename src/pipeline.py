import numpy as np
import sys
sys.path.append('/home/fberendse/git/galvanize/work/capstone_1/src/')
from imputationtypes import ImputationTypes as it
from cohorttypes import CohortTypes as ct
from ipedscollection import IpedsCollection
from sklearn.preprocessing import StandardScaler 
# from ipedsdatabase import IpedsDatabase


def grad_make_pcts(df, class_sum, classes, dropna=False, 
                   replace=False, alpha=0):
    '''
    Computes percentages for the graduation rates columns.
    Laplace smoothing is applied to avoid divison by zero
    for categories without students if alpha is nonzero.
    '''
    l2cols = ['gr2mort', 'grasiat', 'grbkaat', 'grhispt', 'grwhitt']
    l2baseline = 'grwhitt'
    l2cols.remove(l2baseline)
    l2cols.insert(0, l2baseline)

    # calculate the count over all races/ethnicities
    n_all_classes = df.loc[:, class_sum].sum(axis=1)

    new_cols = []
    for cl in classes:
        # calculate the class count over all races
        n_class = df.loc[:, cl].sum(axis=1)
        pct_baseline = (str(cl)+'_pct', l2baseline)
        for col in l2cols:
            pct_col = (str(cl)+'_pct', col)
            new_cols.append(pct_col)
            df[pct_col] = round(
                    (df[(cl, col)]+alpha*n_class)*100 /
                    (df[(class_sum, col)]+alpha*n_all_classes),
                    0)
            if col == l2baseline:
                no_baseline_mask = df[pct_baseline] == 0
            else:
                rat_col = (str(cl)+'_rat', col)
                new_cols.append(rat_col)
                df[rat_col] = round(df[pct_col] / df[pct_baseline], 2)
                df.loc[no_baseline_mask, rat_col] = np.nan
            if replace:
                df.drop((cl, col), axis=1, inplace=True)
    if dropna:
        df.dropna(axis=0, how='any', subset=new_cols, inplace=True)
    if replace:
        for col in l2cols:
            df.drop((class_sum, col), axis=1, inplace=True)
    return


def grad_ps_make_pcts(df, cat_partials, cat_totals, dropna=False,
                      replace=False, alpha=0):
    '''
    Computes percentages for the pell grant/SSL graduation rates columns.
    Laplace smoothing is applied to avoid divison by zero for categories
    without students if alpha is nonzero.
    '''

    baseline = 'nrcmbac'
    idx = cat_partials.index(baseline)
    val = cat_partials.pop(idx)
    cat_partials.insert(0, val)
    val = cat_totals.pop(idx)
    cat_totals.insert(0, val)

    sum_partials = df.loc[:, partials].sum(axis=1)
    sum_totals = df.loc[:, totals].sum(axis=1)
    new_cols = []
    for p, t in zip(cat_partials, cat_totals):
        new_cols.append(p+'_pct')
        df[p+'_pct'] = (df[p]+alpha*sum_partials)*100/(df[t]+alpha*sum_totals)
        if replace:
            df.drop(p, axis=1, inplace=True)
        if p == baseline:
            no_baseline_mask = df[baseline+'_pct'] == 0
        else:
            new_cols.append(p+'_rat')
            df[p+'_rat'] = round(df[p+'_pct'] / df[baseline+'_pct'], 2)
            df.loc[no_baseline_mask, p+'_rat'] = np.nan

    if dropna:
        df.dropna(axis=0, how='any', subset=new_cols, inplace=True)
    return


def grad_ps_make_ratios():
    pass


def calc_percentile(table, cols, avg_col=None):
    '''
    Calculates the percentile of each column in list cols.
    Returns the average of those percentiles in a column named avg_col 
    when set to a string.
    '''
    df = table.df
    for c in cols:
        df[c+'_pctl'] = df[c].rank(method='max', pct=True)*100
    if type(avg_col) is str:
        pctl_cols = [c+'_pctl' for c in cols]
        df[avg_col] = df.loc[:, pctl_cols].mean(axis=1)
    return


def standardize(table, cols, avg_col=None):
    '''
    scales each column in list cols.
    Returns a column of averages for each row if avg_col
    is set to a string
    '''
    df = table.df
    sc = StandardScaler()
    for c in cols:
        vals = df[c].values.reshape(-1, 1)
        df[c+'_scl'] = sc.fit_transform(vals).ravel()
    if type(avg_col) is str:
        scl_cols = [c+'_scl' for c in cols]
        df[avg_col] = df.loc[:, scl_cols].mean(axis=1)
    return


if __name__ == "__main__":

    exclude_list = [it.data_not_usable,
                    it.do_not_know,
                    it.left_blank,
                    it.not_applicable]

    tc = IpedsCollection()

    # HD2017 table
    hd_keep = ['unitid', 'instnm', 'city', 'stabbr', 'iclevel', 'control',
               'hloffer', 'hbcu', 'tribal', 'locale', 'instsize', 'longitud',
               'latitude', 'landgrnt']
    hd_map_values = {'iclevel': ct.hd_iclevel_str,
                     'control': ct.hd_control_str,
                     'hloffer': ct.hd_hloffer_str,
                     'hbcu': ct.hd_hbcu_str,
                     'tribal': ct.hd_tribal_str,
                     'locale': ct.hd_locale_str,
                     'instsize': ct.hd_instsize_str,
                     'landgrnt': ct.hd_landgrnt_str}
    hd_cat_cols = ['iclevel', 'control', 'hloffer', 'hbcu', 'tribal', 'locale',
                   'instsize', 'landgrnt']
    tc.update_meta('hd2017',
                   filepath='data/hd2017.csv',
                   keep_columns=hd_keep,
                   map_values=hd_map_values,
                   category_columns=hd_cat_cols,
                   exclude_imputations=exclude_list)

    # ADM2017 table
    adm_keep = ['unitid', 'applcn', 'admssn', 'enrlt', 'enrlft',
                'satvr25', 'satvr75', 'satmt25', 'satmt75',
                'acten25', 'acten75', 'actmt25', 'actmt75']
    tc.update_meta('adm2017',
                   filepath='data/adm2017.csv',
                   keep_columns=adm_keep,
                   exclude_imputations=exclude_list)

    # GR2017 table
    gr_keep = ['unitid', 'chrtstat', 'cohort', 'grasiat',
               'grbkaat', 'grhispt', 'grwhitt', 'gr2mort']
    gr_col_levels = ['chrtstat']
    gr_map_values = {'chrtstat': ct.gr_chrtstat_str,
                     'cohort': ct.gr_cohort_str}
    gr_filter_values = {'chrtstat': ['cstrevex', 'cstcball'],
                        'cohort': ['cobach']}
    tc.update_meta('gr2017',
                   filepath='data/gr2017.csv',
                   keep_columns=gr_keep,
                   map_values=gr_map_values,
                   filter_values=gr_filter_values,
                   col_levels=gr_col_levels,
                   exclude_imputations=exclude_list)

    # GR2017_PELL_SSL table
    pell_ssl_keep = ['unitid', 'psgrtype', 'pgadjct', 'pgcmbac', 'ssadjct',
                     'sscmbac', 'nradjct', 'nrcmbac']
    grp_map_values = {'psgrtype': ct.grp_psgrtype_str}
    grp_filter_values = {'psgrtype': 'psgrbac'}
    tc.update_meta('gr2017_pell_ssl',
                   filepath='data/gr2017_pell_ssl.csv',
                   keep_columns=pell_ssl_keep,
                   map_values=grp_map_values,
                   filter_values=grp_filter_values,
                   exclude_imputations=exclude_list)

    # SFA2017 table
    sfa_keep = ['unitid', 'uagrntp', 'upgrntp', 'grntn2', 'grnton2', 'grntwf2',
                'grntof2', 'npgrn2', 'grnt4a2']
    tc.update_meta('sfa2017',
                   filepath='data/sfa1617.csv',
                   keep_columns=sfa_keep,
                   exclude_imputations=exclude_list)

    # Process all of the tables, but do not merge
    tc.import_all()
    print(tc.get_row_counts())
    tc.clean_all()
    print(tc.get_row_counts())
    tc.map_values_all()
    tc.filter_all()
    print(tc.get_row_counts())
    tc.encode_columns_all()
    print(tc.get_row_counts())
    tc.make_multicols_all()
    print(tc.get_row_counts())

    print("Calculating percentiles")
    adm = tc.meta['adm2017']['table']
    partials = ['enrlft', 'enrlt', 'admssn']
    totals = ['enrlt', 'admssn', 'applcn']
    adm.make_pct_columns(partials, totals, replace=True, dropna=False)
    standardize(adm, ['satvr25', 'acten25'], avg_col='en25')
    standardize(adm, ['satvr75', 'acten75'], avg_col='en75')
    standardize(adm, ['satmt25', 'actmt25'], avg_col='mt25')
    standardize(adm, ['satmt75', 'actmt75'], avg_col='mt75')

    # # Comment this block out for EDA
    # drop_list = ['satvr25', 'satvr25_scl', 'satvr75', 'satvr75_scl',
    #              'satmt25', 'satmt25_scl', 'satmt75', 'satmt75_scl',
    #              'acten25', 'acten25_scl', 'acten75', 'acten75_scl',
    #              'actmt25', 'actmt25_scl', 'actmt75', 'actmt75_scl']
    # adm.df.drop(drop_list, axis=1, inplace=True)

    gr = tc.meta['gr2017']['table']
    chrtstats = ['cstrevex', 'cstcball']
    for cs in chrtstats:
        gr.df.drop((cs, 'cohort'), axis=1, inplace=True)

    # Calculate percentages and ratios for the graduation rate table
    gr.df.fillna(0, inplace=True)
    _ = chrtstats.pop(0)
    # in the following line, make replace=False for eda; true for modeling
    grad_make_pcts(gr.df, 'cstrevex', chrtstats, dropna=False,
                   replace=False, alpha=0.01)
    gr.df.sort_index(level=0, axis=1, inplace=True)

    # Calculate percentages and ratios for the PELL/SSL table
    grp = tc.meta['gr2017_pell_ssl']['table']
    partials = ['pgcmbac', 'sscmbac', 'nrcmbac']
    totals = ['pgadjct', 'ssadjct', 'nradjct']
    grad_ps_make_pcts(grp.df, partials, totals, dropna=False,
                      replace=True, alpha=0.01)
    grp.df.drop('psgrtype', axis=1, inplace=True)

    # Calculate percentages for the student financial aid table
    sfa = tc.meta['sfa2017']['table']
    mask = sfa.df['grntn2'].isnull()
    sfa.df = sfa.df[~mask]
    sfa.df.fillna(0, inplace=True)
    partials = ['grnton2', 'grntwf2', 'grntof2']
    totals = ['grntn2', 'grntn2', 'grntn2']
    sfa.make_pct_columns(partials, totals, replace=True, dropna=False)
    print(tc.get_row_counts())

    tc.merge_all()
    mdf = tc.merged_table.df
    mdf.dropna(how='any', inplace=True)
    print(mdf.info(max_cols=150))

    new_col_names = []
    for c in mdf.columns:
        new_name = c[0]+'_'+c[1] if type(c) is tuple else c
        new_col_names.append(new_name)
    mdf.columns = new_col_names
    print("writing to file")
    tc.merged_table.df.set_index('unitid')
    tc.merged_table.write_csv('data/ipeds_2017_cats_eda.csv')
