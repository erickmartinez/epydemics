# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:55:40 2020

@author: Erick
"""
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import svd
import matplotlib.gridspec as gridspec
import os
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import EngFormatter
from epydemics import seir_model, confidence as cf
import datetime
from scipy.optimize import OptimizeResult

# Source https://ourworldindata.org/coronavirus-source-data
# csv_data = './full_data.csv'
csv_data = 'https://covid.ourworldindata.org/data/full_data.csv'
# Source: https://worldpopulationreview.com/
csv_population = './population_by_country.csv'

results_folder = './SEIR'
csv_results = 'fitting_results.csv'

df_confd = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
df_death = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
df_recvd = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')

data_color = 'C0'
location = 'Italy'

add_days = 365

plot_pbands = True
xpos = -100
ypos = 50

start_idx = 15
before_day = 0

xfmt = ScalarFormatter(useMathText=True)
xfmt.set_powerlimits((-3, 3))
engfmt = EngFormatter(places=1, sep=u"\N{THIN SPACE}")  # U+2009

if location == 'United States':
    alt_loc = 'US'
elif location == 'South Korea':
    alt_loc = 'Korea, South'
else:
    alt_loc = location

removed_color = 'C4'


def fobj_seir(p: np.ndarray, time: np.ndarray, infected_: np.ndarray,
              removed_: np.ndarray, population_: float, I0_: int, R0_: int = 0):
    p = np.power(10, p)
    sol = seir_model.seir_model(time, N=population_, beta=p[0], gamma=p[1],
                                sigma=p[2], I0=I0_, R0=R0_, E0=p[3])
    y = sol.sol(time)
    S, E, I, R = y

    n = len(infected_)
    residual = np.empty(n * 2)
    for j in range(n):
        residual[j] = I[j] - infected_[j]  # np.log10(I[i]+1) - np.log10(infected[i]+1)
        residual[j + n] = R[j] - removed_[j]  # np.log10(R[i]+1) - np.log10(removed[i]+1)
    return residual


def seir(time: np.ndarray, p: np.ndarray, population_: float, I0_: int, R0_: int = 0):
    p = np.power(10, p)
    sol = seir_model.seir_model(time, N=population_, beta=p[0], gamma=p[1],
                                sigma=p[2], I0=I0_, R0=R0_, E0=p[3])
    y = sol.sol(time)
    S, E, I, R = y
    points = len(time)
    result = np.zeros((points, 2), dtype=np.float)
    for n, j, r in zip(range(points), I, R):
        result[n] = (j, r)
    return result


defaultPlotStyle = {'font.size': 14,
                    'font.family': 'Arial',
                    'font.weight': 'regular',
                    'legend.fontsize': 14,
                    'mathtext.fontset': 'stix',
                    #                    'mathtext.rm': 'Times New Roman',
                    #                    'mathtext.it': 'Times New Roman:italic',#'Arial:italic',
                    #                    'mathtext.cal': 'Times New Roman:italic',#'Arial:italic',
                    #                    'mathtext.bf': 'Times New Roman:bold',#'Arial:bold',
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.major.size': 4.5,
                    'xtick.major.width': 1.75,
                    'ytick.major.size': 4.5,
                    'ytick.major.width': 1.75,
                    'xtick.minor.size': 2.75,
                    'xtick.minor.width': 1.0,
                    'ytick.minor.size': 2.75,
                    'ytick.minor.width': 1.0,
                    'ytick.right': False,
                    'lines.linewidth': 2.5,
                    'lines.markersize': 10,
                    'lines.markeredgewidth': 0.85,
                    'axes.labelpad': 5.0,
                    'axes.labelsize': 16,
                    'axes.labelweight': 'regular',
                    'axes.linewidth': 1.25,
                    'axes.titlesize': 16,
                    'axes.titleweight': 'bold',
                    'axes.titlepad': 6,
                    'figure.titleweight': 'bold',
                    'figure.dpi': 100}


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except Exception as e:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def latex_format(x: float, digits: int = 2):
    if np.isinf(x):
        return r'$\infty$'
    digits = abs(digits)
    fmt_dgts = '%%.%df' % digits
    fmt_in = '%%.%dE' % digits
    x_str = fmt_in % x
    x_sci = (np.array(x_str.split('E'))).astype(np.float)
    if abs(x_sci[1]) <= 3:
        fmt_dgts = '%%.%df' % digits
        return fmt_dgts % x
    if digits == 0:
        return r'$\mathregular{10^{%d}}$' % x_sci[1]
    else:
        ltx_str = fmt_dgts % x_sci[0]
        ltx_str += r'$\mathregular{\times 10^{%d}}$' % x_sci[1]
        return ltx_str


covid_type = np.dtype([('date', 'M8[ns]'),
                       ('confirmed', 'u8'),
                       ('recovered', 'u8'),
                       ('dead', 'u8'),
                       ('infected', 'u8')])

if __name__ == '__main__':

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    column_msk = np.zeros(len(df_confd.columns), dtype=bool)
    for i in range(len(df_confd.columns)):
        if i == 0 or i > 3:
            column_msk[i] = True

    df_confd_country = df_confd[df_confd['Country/Region'] == alt_loc].groupby("Country/Region").sum()
    df_confd_country = df_confd_country[df_confd_country.columns[2::]].T.reset_index()
    df_confd_country = df_confd_country.rename(columns={alt_loc: 'confirmed',
                                                        'index': 'date'})

    df_death_country = df_death[df_death['Country/Region'] == alt_loc].groupby("Country/Region").sum()
    df_death_country = df_death_country[df_death_country.columns[2::]].T.reset_index()
    df_death_country = df_death_country.rename(columns={alt_loc: 'dead',
                                                        'index': 'date'})

    df_recvd_country = df_recvd[df_recvd['Country/Region'] == alt_loc].groupby("Country/Region").sum()
    df_recvd_country = df_recvd_country[df_recvd_country.columns[2::]].T.reset_index()
    df_recvd_country = df_recvd_country.rename(columns={alt_loc: 'recovered',
                                                        'index': 'date'})

    df_full = pd.merge(df_confd_country, df_recvd_country,
                       how="outer").fillna(0)

    df_full = pd.merge(df_full, df_death_country,
                       how="outer").fillna(0)

    df_full = df_full.eval('infected = confirmed - recovered - dead')
    df_full['date'] = pd.to_datetime(df_full['date'], format='%m/%d/%y')

    df_full = df_full[df_full['infected'] > 0]

    df = pd.read_csv(csv_data)
    df = df[df['location'] == location]
    covid_by_country = df.sort_values(by=['date'])
    #    covid_by_country = covid_by_country[0:-1]
    cv_date = pd.to_datetime(df_full['date']).to_numpy()
    time_s = 1E-9 * (cv_date - np.amin(cv_date))
    time_days = np.array([t / 86400 for t in time_s], dtype=float)

    df_population = pd.read_csv(csv_population)
    population = float(df_population[df_population['name'] == location]['pop2020']) * 1000

    infected = df_full['infected'].to_numpy()
    recovered = df_full['recovered'].to_numpy()
    confirmed = df_full['confirmed'].to_numpy()
    dead = df_full['dead'].to_numpy()
    #    new_cases = covid_by_country['new_cases'].to_numpy()

    I0 = infected[0]
    id_start = np.argmin(infected <= I0) - 2
    range_msk = np.zeros_like(confirmed, dtype=bool)
    for i, v in enumerate(confirmed):
        if i >= id_start:
            range_msk[i] = True

    time_days = time_days[range_msk]
    infected = infected[range_msk]
    recovered = recovered[range_msk]
    confirmed = confirmed[range_msk]
    dead = dead[range_msk]
    cv_date = cv_date[range_msk]
    removed = dead + recovered

    time_days_fit = time_days[start_idx::]
    infected_fit = infected[start_idx::]
    recovered_fit = recovered[start_idx::]
    confirmed_fit = confirmed[start_idx::]
    dead_fit = dead[start_idx::]
    cv_date_fit = cv_date[start_idx::]
    I0_fit = infected_fit[0]

    removed_fit = dead_fit + recovered_fit
    R0_fit = removed_fit[0]

    #    new_cases = new_cases[idx]
    if before_day > 0:
        idx = time_days_fit < before_day
        time_days_fit = time_days_fit[idx]
        infected_fit = infected_fit[idx]
        recovered_fit = recovered_fit[idx]
        confirmed_fit = confirmed_fit[idx]
        dead_fit = dead_fit[idx]
        cv_date_fit = cv_date_fit[idx]
        removed_fit = removed_fit[idx]
        I0_fit = infected_fit[0]

    t0 = datetime.datetime.utcfromtimestamp(cv_date[0].astype(datetime.datetime) * 1E-9)
    n_days = len(time_days)

    all_tol = np.finfo(np.float64).eps
    res: OptimizeResult = optimize.least_squares(fobj_seir, np.log10(np.array([1E-1, 1E-2, 1E0, I0_fit])),
                                                 jac='3-point',
                                                 # bounds=np.log10([1E-50,1E-50,1E-15,1], [np.inf, np.inf,np.inf, np.inf]),
                                                 args=(time_days_fit, infected_fit, removed_fit,
                                                       population, I0_fit, R0_fit),
                                                 xtol=all_tol * 1,
                                                 ftol=all_tol * 1,
                                                 gtol=all_tol * 1,
                                                 # x_scale='jac',
                                                 # loss='soft_l1', f_scale=0.1,
                                                 # loss='cauchy', f_scale=0.1,
                                                 #  max_nfev=n_days*1000,
                                                 verbose=2)

    popt_log = res.x
    popt = np.power(10, popt_log)

    xpred = np.linspace(np.amin(time_days_fit), np.amax(time_days_fit), 100)
    xpred_2 = np.linspace(np.amin(time_days_fit), np.amax(time_days_fit) + add_days, len(time_days) + add_days)
    ypred_2 = seir(xpred_2, popt_log, population, I0_fit)

    ysize = len(res.fun)
    cost = 2 * res.cost  # res.cost is half sum of squares!
    s_sq = cost / (ysize - popt.size)

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov_log = np.dot(VT.T / s ** 2, VT)
    pcov_log = pcov_log * s_sq

    #    print(pcov_log)

    if pcov_log is None:
        # indeterminate covariance
        print('Failed estimating pcov')
        pcov_log = np.zeros((len(pcov_log), len(pcov_log)), dtype=float)
        pcov_log.fill(np.inf)

    #    pcov = np.power(10, pcov_log)

    ci_seir = np.power(10, cf.confint(ysize, popt_log, pcov_log))


    def smodel(x, p):
        return seir(x, p, population, I0_fit, R0_fit)


    ypred_seir, lpb_seir, upb_seir = cf.predint_multi(xpred, time_days_fit, infected_fit, smodel, res,
                                                      mode='observation')

    R0_cal = popt[0] / popt[1]

    mpl.rcParams.update(defaultPlotStyle)

    fig = plt.figure()
    fig.set_size_inches(5.0, 10, forward=True)
    fig.subplots_adjust(hspace=0.5, wspace=0.15)
    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1])
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1,
                                            subplot_spec=gs0[0])

    ax1 = fig.add_subplot(gs00[0, 0])
    ax2 = fig.add_subplot(gs00[1, 0])
    ax3 = fig.add_subplot(gs00[2, 0])

    model2_color = data_color
    if isinstance(data_color, str):
        data_color = mpl.colors.to_rgb(data_color)
        removed_color = mpl.colors.to_rgb(removed_color)
        model2_color = mpl.colors.to_rgb(data_color)

    pband_color = lighten_color(data_color, 0.25)
    pband_color2 = lighten_color(removed_color, 0.25)

    ax1.plot(time_days, confirmed, 'o', color='tab:orange', fillstyle='full', label='total')
    ax1.plot(time_days, recovered, 's', color='tab:green', fillstyle='none', label='recovered')
    ax1.plot(time_days, dead, '^', color='tab:red', fillstyle='none', label='dead')
    ax1.plot(time_days, infected, 'o', color=data_color, fillstyle='none', label='infected')

    if plot_pbands:
        ax2.fill_between(xpred, lpb_seir[:, 0], upb_seir[:, 0], color=pband_color)
        ax2.fill_between(xpred, lpb_seir[:, 1], upb_seir[:, 1], color=pband_color2)

    ax2.plot(time_days_fit, infected_fit, 'o', color=data_color, fillstyle='none', label='infected')
    ax2.plot(time_days_fit, removed_fit, 'v', color=removed_color, fillstyle='none', label='removed')
    ax2.plot(xpred, ypred_seir[:, 0], color=model2_color)
    ax2.plot(xpred, ypred_seir[:, 1], color=removed_color)

    #    ax2_b = ax2.twinx()
    #    ax3.tick_params(axis='y', labelcolor='C1')

    ax3.plot(xpred_2, ypred_2[:, 0], ls='-', color=data_color, label='infected')
    ax3.plot(xpred_2, ypred_2[:, 1], ls='--', color=removed_color, label='removed')
    infected_res = ypred_2[:, 0]
    removed_res = ypred_2[:, 1]
    imax_seir = np.amax(infected_res)
    rmax_seir = np.amax(removed_res)
    oymax = max(imax_seir, rmax_seir)
    tmax_seir = int(xpred_2[infected_res == imax_seir])

    peak_date = t0 + datetime.timedelta(days=tmax_seir)

    peak_str = 'Date = {0}\nCases: {1}'.format(peak_date.strftime('%Y/%m/%d'),
                                               engfmt.format_data(imax_seir))

    ax3.annotate(peak_str,
                 xy=(tmax_seir, imax_seir), xycoords='data',
                 xytext=(xpos, ypos), textcoords='offset points',
                 color='tab:red',
                 fontsize=12,
                 arrowprops=dict(arrowstyle="->",
                                 color='tab:red',
                                 connectionstyle="angle3,angleA=0,angleB=90"))

    leg_colors = ['tab:orange', 'tab:green', 'tab:red', 'C0']
    leg1 = ax1.legend(loc='upper left', frameon=False)
    for i, text in enumerate(leg1.get_texts()):
        text.set_color(leg_colors[i])

    leg_colors2 = [data_color, removed_color]
    leg2 = ax2.legend(loc='upper left', frameon=False)
    for i, text in enumerate(leg2.get_texts()):
        text.set_color(leg_colors2[i])

    ax1.set_xlabel('Time (days since {0})'.format(t0.strftime('%Y/%m/%d')))
    ax1.set_ylabel('Cases (#)')

    ax2.set_xlabel('Time (days since {0})'.format(t0.strftime('%Y/%m/%d')))
    ax2.set_ylabel('Cases (#)')

    ax3.set_xlabel('Time (days since {0})'.format(t0.strftime('%Y/%m/%d')))
    ax3.set_ylabel('Cases (#)')  # , color='C1'

    ax1.set_title('COVID-19 cases in {0}'.format(location))

    res_str = '$\\beta$ = {0} (days)$^{{-1}}$\n95% CI: [{1},{2}]\n'.format(latex_format(popt[0], 3),
                                                                           latex_format(ci_seir[0][0], 3),
                                                                           latex_format(ci_seir[0][1], 3))
    res_str += '$\\gamma$ = {0} (days)$^{{-1}}$\n95% CI: [{1},{2}]\n'.format(latex_format(popt[1], 3),
                                                                             latex_format(ci_seir[1][0], 3),
                                                                             latex_format(ci_seir[1][1], 3))
    res_str += '$\\sigma$ = {0} (days)$^{{-1}}$\n95% CI: [{1},{2}]\n'.format(latex_format(popt[2], 3),
                                                                             latex_format(ci_seir[2][0], 3),
                                                                             latex_format(ci_seir[2][1], 3))
    #    res_str += '$I_0 =$  {0:.3g}\n95% CI: [{1:.3g},{2:.3g}]'.format(popt_seir[2],
    #                    ci_seir[2][0], ci_seir[2][1])
    res_str += '$I_0 =$  {0:.3g}, $R_0$ = {1:.0f}, $E_0 =$ {2:.2g}, $t_0 =$ {3:.0f}'.format(I0_fit,
                                                                                            R0_fit, popt[3],
                                                                                            time_days_fit[0])

    ax2.text(0.05, 0.1, res_str,
             horizontalalignment='left',
             verticalalignment='bottom',
             transform=ax2.transAxes,
             fontsize=10,
             color='k',
             zorder=6)

    ax3.text(0.05, 0.1, 'Population: {0}\n$\\mathcal{{R}}_0 = $ {1}'.format(
        engfmt(population),
        latex_format(R0_cal, 3)),
             horizontalalignment='left',
             verticalalignment='bottom',
             transform=ax3.transAxes,
             fontsize=11,
             color='k',
             zorder=6)

    #    model_str = '$N = N_0 e^{bt} + N_1$'

    ax2.set_title('SEIR model fit')

    ax3.set_title('Predicted infections')

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim(top=np.amax(ypred_seir[:]) * 100, bottom=1)

    locmaj = mpl.ticker.LogLocator(base=10.0, numticks=5)
    locmin = mpl.ticker.LogLocator(base=10.0, numticks=9, subs=np.arange(0.1, 1.1, 0.1))

    #    ax1.yaxis.set_major_formatter(engfmt)
    #    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax1.yaxis.set_major_locator(locmaj)
    ax1.yaxis.set_minor_locator(locmin)
    ax1.yaxis.set_ticks_position('both')

    ax1.xaxis.set_major_locator(mticker.MaxNLocator(5, prune=None))
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))

    #    ax2.yaxis.set_major_formatter(engfmt)
    #    ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax2.yaxis.set_ticks_position('both')

    #    ax2.xaxis.set_major_locator(mticker.MaxNLocator(6,prune=None))
    #    ax2.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    ax2.yaxis.set_major_locator(locmaj)
    ax2.yaxis.set_minor_locator(locmin)

    ax3.xaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
    ax3.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))

    ax3.yaxis.set_major_formatter(engfmt)
    ax3.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax3.yaxis.set_ticks_position('both')

    #    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.tight_layout()
    plt.show()

    filetag = 'covid19_model_{0}_ir'.format(location)
    if before_day > 0:
        filetag += '_before_{0}d'.format(before_day)

    fig.savefig(os.path.join(results_folder, filetag + '.png'), dpi=600)

    print('beta =\t{0:.3E} 1/days, 95% CI: [{1:.3E}, {2:.3E}] 1/days'.format(
        popt[0], ci_seir[0][0], ci_seir[0][1]))
    print('gamma =\t{0:.3E} 1/days, 95% CI: [{1:.3E}, {2:.3E}] 1/days'.format(
        popt[1], ci_seir[1][0], ci_seir[1][1]))
    print('sigma =\t{0:.3E} 1/days, 95% CI: [{1:.3E}, {2:.3E}] 1/days'.format(
        popt[2], ci_seir[2][0], ci_seir[2][1]))
    print('E0 =\t{0:.3E} 1/days, 95% CI: [{1:.3E}, {2:.3E}] 1/days'.format(
        popt[3], ci_seir[3][0], ci_seir[3][1]))
