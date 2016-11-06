#!/usr/bin/env python
from __future__ import division
from glob import glob
import os

import numpy as np
import pandas as pd
import progressbar as pb

from pummeler.reader import VERSIONS
from pummeler.stats import load_stats, save_stats


def merge_12to14_15(dir_12to14, dir_15, out_dir):
    os.makedirs(out_dir)
    # We'll need to do two feature transformations:
    # - merge RAC_NH and RAC_PI from 15 data into RAC_NHPI
    # - convert 2014 dollars from 12to14 into 2015 dollars
    # - drop some allocation flags

    # Source for this inflation factor: http://www.usinflationcalculator.com
    # Not exactly the source Census uses, but whatever, close enough
    # (and inflation is low enough anyway).
    adjinc = 1.0011869762
    version_a = VERSIONS['2010-14_12-14']
    version_b = VERSIONS['2015']
    to_adjinc = version_a['to_adjinc']
    flags_to_drop = list(
        set(version_b['alloc_flags']) - set(version_a['alloc_flags']))

    stats_a = load_stats(os.path.join(dir_12to14, 'stats.h5'))
    stats_b = load_stats(os.path.join(dir_15, 'stats.h5'))

    stats = {}
    stats['version'] = '2012-15_manual'

    n_a = stats_a['n_total']
    n_b = stats_b['n_total']
    stats['n_total'] = n_a + n_b

    wt_a = stats_a['wt_total']
    wt_b = stats_a['wt_total']

    # want to adjust weights of a so that wt_a / n_a == wt_b / n_b
    weight_mult = (wt_b / n_b) / (wt_a / n_a)
    stats['wt_total'] = weight_mult * wt_a + wt_b

    Ex_a = stats_a['real_means']
    Ex_a[to_adjinc] *= adjinc
    Ex_b = stats_b['real_means']
    stats['real_means'] = Ex = n_a/(n_a+n_b) * Ex_a + n_b/(n_a+n_b) * Ex_b

    stds_a = stats_a['real_stds']
    stds_a[to_adjinc] *= adjinc
    Ex2_a = stds_a ** 2 + Ex_a ** 2
    Ex2_b = stats_b['real_stds'] ** 2 + Ex_b ** 2
    Ex2 = n_a/(n_a+n_b) * Ex2_a + n_b/(n_a+n_b) * Ex2_b
    stats['real_stds'] = (Ex2 - Ex ** 2).map(np.sqrt)

    a_vc = stats_a['value_counts']
    b_vc = stats_b['value_counts']
    b_vc['RACNHPI'] = b_vc.pop('RACPI') + b_vc.pop('RACNH')
    b_vc['RACNHPI'].name = 'RACNHPI'

    stats['value_counts'] = vc = {}
    for k in list(stats_a['value_counts'].keys()):
        vc[k] = a_vc.pop(k) + b_vc.pop(k)
    assert not a_vc
    assert set(b_vc).issubset(flags_to_drop)

    def _change_a(feats_a):
        feats_a[to_adjinc] *= adjinc
        feats_a.PWGTP *= weight_mult

    def _change_b(feats_b):
        feats_b['RACNHPI'] = feats_b['RACNH'] | feats_b['RACPI']
        feats_b['RACNHPI'].name = 'RACNHPI'
        feats_b.drop(['RACNH', 'RACPI'] + flags_to_drop, axis=1, inplace=True)

    _change_a(stats_a['sample'])
    _change_b(stats_b['sample'])
    combo_samp = pd.concat([stats_a['sample'], stats_b['sample']])
    stats['sample'] = combo_samp.iloc[np.random.choice(
        combo_samp.shape[0], replace=False,
        p=combo_samp.PWGTP / combo_samp.PWGTP.sum(),
        size=max(stats_a['sample'].shape[0], stats_b['sample'].shape[0]))]

    save_stats(os.path.join(out_dir, 'stats.h5'), stats)

    for fn in pb.ProgressBar()(glob(os.path.join(dir_12to14, 'feats_*.h5'))):
        bn = os.path.basename(fn)

        a = pd.read_hdf(fn)
        _change_a(a)

        b = pd.read_hdf(os.path.join(dir_15, bn))
        _change_b(b)

        pd.concat([a, b]).to_hdf(
            os.path.join(out_dir, bn), 'df', format='table',
            mode='w', complib='blosc', complevel=6)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_12to14')
    parser.add_argument('dir_15')
    parser.add_argument('out_dir')
    args = parser.parse_args()
    merge_12to14_15(**vars(args))


if __name__ == '__main__':
    main()