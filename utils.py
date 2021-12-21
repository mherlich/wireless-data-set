import os
import numpy as np
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt

def evaluate(p, tv, absval=None):
    col = ['Strategy', 'MAE', 'MAPE', 'MAPEm', 'MAPEn', 'MAPEc', 'MAPEl6', 'MSE', "R2", 'd1%', 'd3%', 'd10%', 'd30%', 'q50%', 'q90%', 'q95%', 'q99%']    
    if absval == None:
        col.extend(['d1Mbit/s', 'd3Mbit/s', 'd10Mbit/s', 'd30Mbit/s'])
    else:
        col.extend(['d'+str(a) for a in absval])
    r = pd.DataFrame(columns=col)
    for c in p:
        d = {
            'Strategy': c,
            'MAE': (p[c]-tv).abs().mean(),
            'MAPE': ((p[c]-tv).abs()/tv).mean(),
            'MAPEm': ((p[c]-tv).abs()/np.max([p[c],tv], axis=0)).mean(),
            'MAPEn': ((p[c]-tv).abs()/np.max([p[c].abs(),tv], axis=0)).mean(),
            'MAPEc': (np.min([np.abs(p[c]-tv)/tv, np.ones(len(tv))], axis=0)).mean(),
            'MAPEl6': ((p[c]-tv).abs()/np.max([tv, np.full(len(tv), 1e6)], axis=0)).mean(),
            'MSE': (np.power(p[c]-tv, 2)).mean(),
            'R2': sklearn.metrics.r2_score(tv, p[c]),
            'd1%': ((p[c]-tv).abs()/tv.abs()<0.01).mean(),
            'd3%': ((p[c]-tv).abs()/tv.abs()<0.03).mean(),
            'd10%': ((p[c]-tv).abs()/tv.abs()<0.10).mean(),
            'd30%': ((p[c]-tv).abs()/tv.abs()<0.30).mean(),
            'q50%': (p[c]-tv).abs().quantile(q=0.5),
            'q90%': (p[c]-tv).abs().quantile(q=0.9),
            'q95%': (p[c]-tv).abs().quantile(q=0.95),
            'q99%': (p[c]-tv).abs().quantile(q=0.99),
        }
        if absval == None:
            for a in [1, 3, 10, 30]:
                d['d'+str(a)+'Mbit/s'] = np.mean(np.abs(p[c]-tv)<a*10**6)
        else:
            for a in absval:
                d['d'+str(a)] = np.mean(np.abs(p[c]-tv) < a)
        r = r.append(d, ignore_index=True)
    return r

def fillna(df, columns, method="median", mark=False):
    dfi = df.copy()
    for i in columns:
        if method == "median":
            dfi[i] = df[i].fillna(df[i].median())
        elif method == "mean":
            dfi[i] = df[i].fillna(df[i].mean())
        elif method == "zero":
            dfi[i] = df[i].fillna(0)
        else:
            raise NameError("Cannot recognize " + method)
        if mark:
            dfi[i+"_na"] = 1 * df[i].isna()
    return dfi

def create_plots(df, log_dir):
    os.mkdir(log_dir)

    # Histogram of true value
    df.tv.hist(bins=101)
    plt.gcf().savefig(log_dir+'/tv_pdf.png')
    plt.close()
    df.tv.hist(cumulative=True, histtype='step', bins=101)
    plt.gcf().savefig(log_dir+'/tv_cdf.png')
    plt.close()

    # Histogram of absolute error
    df.absdiff.hist(bins=101)
    plt.gcf().savefig(log_dir+'/abs_pdf.png')
    plt.close()
    df.absdiff.hist(cumulative=True, histtype='step', bins=101)
    plt.gcf().savefig(log_dir+'/abs_cdf.png')
    plt.close()

    # Histogram of relative error
    df.reldiff.hist(bins=101)
    plt.gcf().savefig(log_dir+'/rel_pdf.png')
    plt.close()
    df.reldiff.hist(cumulative=True, histtype='step', bins=101)
    plt.gcf().savefig(log_dir+'/rel_cdf.png')
    plt.close()

    # Plot of estimate depending on true value
    plt.scatter(df.tv, df.NN, marker='.', alpha=0.1)
    plt.gcf().savefig(log_dir+'/est_scatter.png')
    plt.close()
    df.boxplot("NN", by=round(df.tv/1e6, -1), rot=30)
    plt.suptitle('')
    plt.gca().xaxis.set_label_text('');
    plt.gcf().savefig(log_dir+'/est_box.png')
    plt.close()

    # Plot of absolute error depending on true value
    plt.scatter(df.tv, df.absdiff, marker='.', alpha=0.1)
    plt.gcf().savefig(log_dir+'/abs_scatter.png')
    plt.close()
    df.boxplot("absdiff", by=round(df.tv/1e6, -1), rot=30)
    plt.suptitle('')
    plt.gca().xaxis.set_label_text('');
    plt.gcf().savefig(log_dir+'/abs_box.png')
    plt.close()

    # Plots of relative error depending on true value
    plt.scatter(df.tv, df.reldiff, marker='.', alpha=0.1)
    plt.gcf().savefig(log_dir+'/rel_scatter.png')
    plt.close()
    df.boxplot("reldiff", by=round(df.tv/1e6, -1), rot=30)
    plt.suptitle('')
    plt.gca().xaxis.set_label_text('');
    plt.gcf().savefig(log_dir+'/rel_box.png')
    plt.close()

    # Plot of estimate depending on long
    plt.scatter(df.long, df.NN, marker='.', alpha=0.1)
    plt.gcf().savefig(log_dir+'/pos_est_scatter.png')
    plt.close()
    df.boxplot("NN", by=df.long.round(2), rot=30)
    plt.suptitle('')
    plt.gca().xaxis.set_label_text('');
    plt.gcf().savefig(log_dir+'/pos_est_box.png')
    plt.close()

    # Plot of absolute error depending on long
    plt.scatter(df.long, df.absdiff, marker='.', alpha=0.1)
    plt.gcf().savefig(log_dir+'/pos_abs_scatter.png')
    plt.close()
    df.boxplot("absdiff", by=df.long.round(2), rot=30)
    plt.suptitle('')
    plt.gca().xaxis.set_label_text('');
    plt.gcf().savefig(log_dir+'/pos_abs_box.png')
    plt.close()

    # Plots of relative error depending on long
    plt.scatter(df.long, df.reldiff, marker='.', alpha=0.1)
    plt.gcf().savefig(log_dir+'/pos_rel_scatter.png')
    plt.close()
    df.boxplot("reldiff", by=df.long.round(2), rot=30)
    plt.suptitle('')
    plt.gca().xaxis.set_label_text('');
    plt.gcf().savefig(log_dir+'/pos_rel_box.png')
    plt.close()
