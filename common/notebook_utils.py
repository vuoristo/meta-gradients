import numpy as onp
import pandas as pd
from ray.tune import Analysis


def load_data(logdirs):
    all_dfs = []
    all_hypers = []
    for logdir in logdirs:
        analysis = Analysis(logdir)
        configs = analysis.get_all_configs()
        dfs = analysis.trial_dataframes
        for key, df in dfs.items():
            config = pd.json_normalize(configs[key], sep='_').to_dict(orient='records')[0]
            for hyper, value in config.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        df['HYPER_{}_{}'.format(hyper, k)] = v if not isinstance(v, list) else str(v)
                else:
                    df['HYPER_' + hyper] = value if not isinstance(value, list) else str(value)
            df['EXP_DIR'] = logdir
            df['trial_path'] = key
            all_dfs.append(df)
        hypers = list(list(configs.values())[0].keys())
        all_hypers.extend(hypers)
    DF = pd.concat(all_dfs)
    HYPERS = list(set(all_hypers))
    return DF, HYPERS


def get_ckpt_paths(logdirs):
    all_dfs = []
    for logdir in logdirs:
        analysis = Analysis(logdir)
        configs = analysis.get_all_configs()
        dfs = analysis.trial_dataframes
        for key in dfs.keys():
            config = configs[key]
            hyp_config = {}
            for hyper, value in config.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        hyp_config['HYPER_{}_{}'.format(hyper, k)] = v if not isinstance(v, list) else str(v)
                else:
                    hyp_config['HYPER_' + hyper] = value if not isinstance(value, list) else str(value)

            paths = analysis.get_trial_checkpoints_paths(key)
            if len(paths) > 0:
                ps, ckpts = list(zip(*paths))
                df = pd.DataFrame({'path': ps, 'checkpoint': ckpts})
                for k, v in hyp_config.items():
                    df[k] = v
                all_dfs.append(df)
                df['EXP_DIR'] = logdir
    DF = pd.concat(all_dfs)
    return DF


def seed_mavg(data, xseq, **params):
    """
    Fit moving average across multiple seeds
    """
    window = params['method_args']['window']

    smoothed = data.groupby(data.seed).rolling(window).mean().groupby('x')
    y = smoothed.mean()['y'].to_numpy()
    stderr = smoothed.std()['y'].fillna(0.).to_numpy() / onp.sqrt(len(data.seed.unique()))
    x = smoothed.min().index
    out = pd.DataFrame({'x': x, 'y': y})
    out.reset_index(inplace=True, drop=True)

    if params['se']:
        out['ymin'] = out['y'] - stderr
        out['ymax'] = out['y'] + stderr
        out['se'] = stderr

#     if params['se']:
#         out['ymin'], out['ymax'] = gg.stats.smoothers.tdist_ci(
#             y, None, stderr, params['level'])
#         out['se'] = stderr
    return out


def parse_str_col(x, col='value_errors'):
    try:
        values = onp.array(eval(x[col]))
    except:
        values = onp.nan
    return values


def explode(df, lst_cols, fill_value='', preserve_index=False):
    """ from https://stackoverflow.com/a/40449726
    """
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, onp.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = onp.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:onp.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:onp.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res


def softplus(x):
    return onp.logaddexp(x, 0)
