def collapse_onehots(df, features, default, val_dict=None):
    root = features[0].split('_')[0]
    df_part = df.loc[:, features]
    if df_part.sum(axis=1)[0]:
        col = df_part.idxmax(axis=1)[0]
        val = col.split(root+'_')[1]
    else:
        val = default
    if val_dict is not None:
        df[root] = val_dict[val]
    else:
        df[root] = val
    df.drop(columns=features, inplace=True)


def collapse_all_onehots(df):
    features = ['control_public', 'control_privnp', 'control_na']
    default = 'privfp'
    val_dict = {'public': 'Public',
                'privnp': 'Private, not-for-profit',
                'privfp': 'Private, for-profit',
                'na': 'Not applicable'}
    collapse_onehots(df, features, default, val_dict)

    features = ['hloffer_assoc', 'hloffer_doct', 'hloffer_bach',
                'hloffer_mast', 'hloffer_2to4yr', 'hloffer_0to1yr',
                'hloffer_postmc', 'hloffer_na', 'hloffer_postbc']
    default = '1to2yr'
    val_dict = {"0to1yr": "Award of less than one academic year",
                "1to2yr": "Award at least 1, but less than 2 academic yrs",
                "assoc": "Associate's degree",
                "2to4yr": "Award at least 2, but less than 4 academic yrs",
                "bach": "Bachelor's degree",
                "postbc": "Postbaccalaureate certificate",
                "mast": "Master's degree",
                "postmc": "Post-master's certificate",
                "doct": "Doctoral degree",
                "na": "Not available"}
    collapse_onehots(df, features, default, val_dict)

    features = ['hbcu_yes']
    default = 'no'
    val_dict = {"yes": "Yes",
                "no": "No"}
    collapse_onehots(df, features, default, val_dict)

    features = ['locale_ctylrg', 'locale_ctysml', 'locale_ctymid',
                'locale_twndst', 'locale_rurfrg', 'locale_twnrem',
                'locale_submid', 'locale_subsml', 'locale_twnfrg',
                'locale_rurdst', 'locale_rurrem']
    default = 'sublrg'
    val_dict = {'ctylrg': "City: Large",
                'ctymid': "City: Midsize",
                'ctysml': "City: Small",
                'sublrg': "Suburb: Large",
                'submid': "Suburb: Midsize",
                'subsml': "Suburb: Small",
                'twnfrg': "Town: Fringe",
                'twndst': "Town: Distant",
                'twnrem': "Town: Remote",
                'rurfrg': "Rural: Fringe",
                'rurdst': "Rural: Distant",
                'rurrem': "Rural: Remote",
                'na': "Not available"}
    collapse_onehots(df, features, default, val_dict)

    features = ['instsize_1to5k', 'instsize_5to10k', 'instsize_10to20k',
                'instsize_gt20k', 'instsize_norpt', 'instsize_na']
    default = '0to1k'
    val_dict = {'0to1k': "Under 1,000",
                '1to5k': "1,000 - 4,999",
                '5to10k': "5,000 - 9,999",
                '10to20k': "10,000 - 19,999",
                'gt20k': "20,000 and above",
                'norpt': "Not reported",
                'na': "Not available"}
    collapse_onehots(df, features, default, val_dict)


def get_features_df(df):

    feat_cols = ['control', 'hloffer', 'hbcu', 'locale', 'instsize',
                 'longitud', 'latitude', 'admssn_pct', 'enrlt_pct',
                 'enrlft_pct', 'en25', 'uagrntp', 'upgrntp', 'npgrn2',
                 'grntof2_pct', 'grntwf2_pct']
    features_df = df.loc[:, feat_cols]
    features_df = features_df.round({'latitude': 4,
                                     'longitud': 4,
                                     'admssn_pct': 0,
                                     'enrlt_pct': 0,
                                     'enrlft_pct': 0,
                                     'en25': 2,
                                     'uagrntp': 0,
                                     'npgrntp': 0,
                                     'npgrn2': 2,
                                     'grntof2_pct': 0,
                                     'grntwf2_pct': 0})

    feat_desc = ["Control", "Highest Award", "HBCU", "Locale", "Size",
                 "Longitude", "Latitude", "Percent Admitted",
                 "Percent Enrolled", "Percent Full Time",
                 "Scaled English 25th Percentile",
                 "Percent Receiving Aid (any)", "Percent Receiving Pell Grant",
                 "Net Price for Pell Grant Recipient",
                 "Percent Living Off Campus",
                 "Percent Living With Family"]
    feat_dict = dict(zip(feat_cols, feat_desc))
    features_df.rename(columns=feat_dict, inplace=True)
    return features_df
