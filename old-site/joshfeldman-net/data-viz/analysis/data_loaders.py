import pandas as pd
import numpy as np

def zillow_pipeline(path, year_moved_start, year_moved_end, region_type):
    assert(region_type in ['county','zipcode','metro'])

    # load data
    df = pd.read_csv(path, encoding = 'mac_roman')
    years = range(year_moved_start, year_moved_end + 1)

    # add region ID
    if region_type == 'county':
        # clean state code
        df.StateCodeFIPS = df.StateCodeFIPS.astype(str)
        df.StateCodeFIPS = df.StateCodeFIPS.apply(
            lambda x: '0'+x if len(x) == 1 else x
        )

        # clean county code
        df.MunicipalCodeFIPS = df.MunicipalCodeFIPS.astype(str)
        df.MunicipalCodeFIPS = df.MunicipalCodeFIPS.apply(
            lambda x: '00' + x if len(x) == 1 else '0' + x if len(x) == 2 else x
        )
        df['region_id'] = df.apply(lambda row: row.StateCodeFIPS+row.MunicipalCodeFIPS, axis = 1)
        df['region_name'] = df['RegionName']
    elif region_type == 'zipcode':
        df['region_id'] = df['RegionName']
        df['region_name'] = df['Metro']
    elif region_type == 'metro':
        df['region_id'] = df['RegionName']
        df['region_name'] = df['RegionName']
    dfs = []
    for year in years:
        for month in range(1,13):
            month_str = '0' + str(month) if month < 10 else str(month)
            year_month_clm = f"{year}-{month_str}"
            try:
                long_df_year_month = (
                    df[[year_month_clm, 'region_id', 'region_name']]
                    .copy()
                    .rename(index=str, columns={
                        year_month_clm: 'median_rent'
                    })
                )
                long_df_year_month['month'] = month
                long_df_year_month['year_moved_start'] = year
                long_df_year_month['year_moved_end'] = year
                long_df_year_month['region_type'] = region_type
                long_df_year_month['MoE'] = np.NaN
                long_df_year_month['source'] = 'zillow'

                dfs.append(long_df_year_month)
            except KeyError:
                pass
    zillow_long_county = pd.concat(dfs)
    return zillow_long_county

def acs_pipeline(path,
                 year_moved_start,
                 year_moved_end,
                 region_type,
                 region_name_clm,
                 recent_median_rent_clm,
                 recent_median_rent_MoE_clm,
                 median_rent_clm,
                 median_rent_MoE_clm,
                 source_name,
                ):
    # load data
    df = pd.read_csv(path, encoding = 'mac_roman', dtype={region_name_clm:str})

    # add region ID
    if region_type == 'county':
        # clean state code
        df.STATEA = df.STATEA.astype(str)
        df.STATEA = df.STATEA.apply(
            lambda x: '0'+x if len(x) == 1 else x
        )

        # clean county code
        df.COUNTYA = df.COUNTYA.astype(str)
        df.COUNTYA = df.COUNTYA.apply(
            lambda x: '00' + x if len(x) == 1 else '0' + x if len(x) == 2 else x
        )
        df['region_id'] = df.apply(lambda row: row.STATEA+row.COUNTYA, axis = 1)
    elif region_type == 'zipcode':
        df['region_id'] = df['ZCTA5A']
    df['region_name'] = df[region_name_clm].astype(str)

    dfs = []
    for name in [f"{source_name}_recent",f"{source_name}_all"]:
        if name == f"{source_name}_recent":
            acs_long = (
                df[[recent_median_rent_clm, recent_median_rent_MoE_clm, 'region_name', 'region_id']]
                .copy()
                .rename(index=str, columns={
                    recent_median_rent_clm: 'median_rent',
                    recent_median_rent_MoE_clm: 'MoE'
                })
            )
            acs_long['source'] = name
            acs_long['year_moved_start'] = year_moved_start
            acs_long['year_moved_end'] = year_moved_end
        else:
            acs_long = (
                df[[median_rent_clm, median_rent_MoE_clm, 'region_name', 'region_id']]
                .copy()
                .rename(index=str, columns={
                    median_rent_clm: 'median_rent',
                    median_rent_MoE_clm: 'MoE'
                })
            )
            acs_long['source'] = name
            acs_long['year_moved_start'] = np.NaN
            acs_long['year_moved_end'] = year_moved_end

        acs_long['month'] = np.NaN
        acs_long['region_type'] = region_type
        dfs.append(acs_long)
    return pd.concat(dfs)
