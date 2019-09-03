import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface


nb_zero = lambda t, mu: (t/(mu+t))**t
zinb_zero = lambda t, mu, p: p + ((1.-p)*((t/(mu+t))**t))
sigmoid = lambda x: 1. / (1.+np.exp(-x))
logit = lambda x: np.log(x + 1e-7) - np.log(1. - x + 1e-7)
tf_logit = lambda x: tf.cast(tf.log(x + 1e-7) - tf.log(1. - x + 1e-7), 'float32')
log_loss = lambda pred, label: np.sum(-(label*np.log(pred+1e-7)) - ((1.-label)*np.log(1.-pred+1e-7)))


def _lrt(ll_full, ll_reduced, df_full, df_reduced):
    # Compute the difference in degrees of freedom.
    delta_df = df_full - df_reduced
    # Compute the deviance test statistic.
    delta_dev = 2 * (ll_full - ll_reduced)
    # Compute the p-values based on the deviance and its expection based on the chi-square distribution.
    pvals = 1. - sp.stats.chi2(delta_df).cdf(delta_dev)

    return pvals


def _fitquad(x, y):
    coef, res, _, _ = np.linalg.lstsq((x**2)[:, np.newaxis] , y-x, rcond=None)
    ss_exp = res[0]
    ss_tot = (np.var(y-x)*len(x))
    r2 = 1 - (ss_exp / ss_tot)
    #print('Coefs:', coef)
    return np.array([coef[0], 1, 0]), r2


def _tf_zinb_zero(mu, t=None):
    a, b = tf.Variable([-1.0], dtype='float32'), tf.Variable([0.0], dtype='float32')

    if t is None:
        t_log = tf.Variable([-10.], dtype='float32')
        t = tf.exp(t_log)

    p = tf.sigmoid((tf.log(mu+1e-7)*a) + b)
    pred = p + ((1.-p)*((t/(mu+t))**t))
    pred = tf.cast(pred, 'float32')
    return pred, a, b, t


def _optimize_zinb(mu, dropout, theta=None):
    pred, a, b, t = _tf_zinb_zero(mu, theta)
    #loss = tf.reduce_mean(tf.abs(tf_logit(pred) - tf_logit(dropout)))
    loss = tf.losses.log_loss(labels=dropout.astype('float32'),
                              predictions=pred)

    optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        optimizer.minimize(sess)
        ret_a = sess.run(a)
        ret_b = sess.run(b)
        if theta is None:
            ret_t = sess.run(t)
        else:
            ret_t = t

    return ret_a, ret_b, ret_t


def plot_mean_dropout(ad, title, ax, opt_zinb_theta=False, legend_out=False):
    expr = ad.X
    mu = expr.mean(0)
    do = np.mean(expr == 0, 0)
    v = expr.var(axis=0)

    coefs, r2 = _fitquad(mu, v)
    theta = 1.0/coefs[0]

    # zinb fit
    coefs = _optimize_zinb(mu, do, theta=theta if not opt_zinb_theta else None)
    print(coefs)

    #pois_pred = np.exp(-mu)
    nb_pred = nb_zero(theta, mu)
    zinb_pred = zinb_zero(coefs[2],
                          mu,
                          sigmoid((np.log(mu+1e-7)*coefs[0])+coefs[1]))

    # calculate log loss for all distr.
    #pois_ll = log_loss(pois_pred, do)
    nb_ll = log_loss(nb_pred, do)
    zinb_ll = log_loss(zinb_pred, do)

    ax.plot(mu, do, 'o', c='black', markersize=1)
    ax.set(xscale="log")

    #sns.lineplot(mu, pois_pred, ax=ax, color='blue')
    sns.lineplot(mu, nb_pred, ax=ax, color='red')
    sns.lineplot(mu, zinb_pred, ax=ax, color='green')

    ax.set_title(title)
    ax.set_ylabel('Empirical dropout rate')
    ax.set_xlabel(r'Mean expression')


    leg_loc = 'best' if not legend_out else 'upper left'
    leg_bbox = None if not legend_out else (1.02, 1.)
    ax.legend(['Genes',
               #r'Poisson $L=%.4f$' % pois_ll,
               r'NB($\theta=%.2f)\ L=%.4f$' % ((1./theta), nb_ll),
               r'ZINB($\theta=%.2f,\pi=\sigma(%.2f\mu%+.2f)) \ L=%.4f$' % (1.0/coefs[2], coefs[0], coefs[1], zinb_ll)],
               loc=leg_loc, bbox_to_anchor=leg_bbox)
    zinb_pval = _lrt(-zinb_ll, -nb_ll, 3, 1)
    print('p-value: %e' % zinb_pval)


def plot_mean_var(ad, title, ax):
    ad = ad.copy()

    sc.pp.filter_cells(ad, min_counts=1)
    sc.pp.filter_genes(ad, min_counts=1)

    m = ad.X.mean(axis=0)
    v = ad.X.var(axis=0)

    coefs, r2 = _fitquad(m, v)

    ax.set(xscale="log", yscale="log")
    ax.plot(m, v, 'o', c='black', markersize=1)

    poly = np.poly1d(coefs)
    sns.lineplot(m, poly(m), ax=ax, color='red')

    ax.set_title(title)
    ax.set_ylabel('Variance')
    ax.set_xlabel(r'$\mu$')

    sns.lineplot(m, m, ax=ax, color='blue')
    ax.legend(['Genes', r'NB ($\theta=%.2f)\ r^2=%.3f$' % (coefs[0], r2), 'Poisson'])

    return coefs[0]


def plot_zeroinf(ad, title, mean_var_plot=False, opt_theta=True):
    if mean_var_plot:
        f, axs = plt.subplots(1, 2, figsize=(15, 5))
        plot_mean_var(ad, title, ax=axs[0])
        plot_mean_dropout(ad, title, axs[1], opt_zinb_theta=opt_theta, legend_out=True)
        plt.tight_layout()
    else:
        f, ax = plt.subplots(1, 1, figsize=(10, 5))
        plot_mean_dropout(ad, title, ax, opt_zinb_theta=opt_theta, legend_out=True)
        plt.tight_layout()
