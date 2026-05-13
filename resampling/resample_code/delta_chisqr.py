import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import os
import matplotlib.patches as patches
import matplotlib as mpl
from scipy.special import gammaincc
from scipy import stats
import seaborn as sns
from scipy.stats import gaussian_kde
import ndtest
from scipy.stats import ks_2samp, norm

def calc_delta_chisqr(comb_halo_df, comb_other_df, other_label, other_marker, other_color, n_iter, save_path=None, plot=False):
    
        
    xlim = (-4.2, -2.0)
    
    # ---PERFORMING THE FITS--- #
    # Fitting polynomial to the halo data (both literature and x-shooter)
    weights_halo = 1 / np.sqrt(comb_halo_df['abund_err']**2 + np.std(comb_halo_df['bestfit_abund_xfe'])**2)
    p_halo = np.polyfit(comb_halo_df['feh'], comb_halo_df['bestfit_abund_xfe'], deg=1, w=weights_halo, cov=False)
    fit_halo = np.poly1d(p_halo)
    xline_halo = np.linspace(xlim[0], xlim[1], 100)
    yline_halo = fit_halo(xline_halo)
    res_halo = comb_halo_df['bestfit_abund_xfe'] - fit_halo(comb_halo_df['feh'])
    std_halo = np.std(res_halo)
    
    # Fitting polynomial to the other data (both literature and x-shooter)
    weights_other = 1 / np.sqrt(comb_other_df['abund_err']**2 + np.std(comb_other_df['bestfit_abund_xfe'])**2)
    p_other = np.polyfit(comb_other_df['feh'], comb_other_df['bestfit_abund_xfe'], deg=1, w=weights_other, cov=False)
    fit_other = np.poly1d(p_other)
    xline_other = np.linspace(xlim[0], xlim[1], 100)
    yline_other = fit_other(xline_other)
    res_other = comb_other_df['bestfit_abund_xfe'] - fit_other(comb_other_df['feh'])
    std_other = np.std(res_other)
    
    
    # ---CALCULATING THE CHI-SQUARE VALUES--- #
    # Calculating the chi-square value for the other data using the halo fit
    # First, calculating the expected values using the halo fit, calculating the chi-square value, reduced chi-square and the p-value
    # Adding the scatter in quadrature to the abundance error
    other_halofit_model = fit_halo(comb_other_df['feh'])
    other_halofit_error = np.sqrt(comb_other_df['abund_err'] ** 2 + std_other ** 2)
    other_halofit_chisqr = np.sum(((comb_other_df['bestfit_abund_xfe'] - other_halofit_model) ** 2) / (other_halofit_error ** 2))
    n_params = 0
    other_halofit_dof = len(comb_other_df) - n_params
    other_halofit_redchisqr = other_halofit_chisqr / other_halofit_dof
    other_halofit_pvalue = gammaincc(other_halofit_dof / 2, other_halofit_chisqr / 2)
    
    # Now doing this for the other fit    
    other_origfit_model = fit_other(comb_other_df['feh'])
    other_origfit_error = np.sqrt(comb_other_df['abund_err'] ** 2 + std_other ** 2)
    other_origfit_chisqr = np.sum(((comb_other_df['bestfit_abund_xfe'] - other_origfit_model) ** 2) / (other_origfit_error ** 2))
    n_params = 2
    other_origfit_dof = len(comb_other_df) - n_params # Number of parameters is the degree
    other_origfit_redchisqr = other_origfit_chisqr / other_origfit_dof
    other_origfit_pvalue = gammaincc(other_origfit_dof / 2, other_origfit_chisqr / 2)
    
    # Calculating the delta chisquare
    delta_chisqr = other_halofit_chisqr - other_origfit_chisqr
    
    # Storing the chi-square values in a dictionary
    chisq_dict = {
        'region': other_label,
        'iter': n_iter,
        'halofit_chisqr': other_halofit_chisqr,
        'halofit_chisqr_red': other_halofit_redchisqr,
        'halofit_pvalue': other_halofit_pvalue,
        'origfit_chisqr': other_origfit_chisqr,
        'origfit_chisqr_red': other_origfit_redchisqr,
        'origfit_pvalue': other_origfit_pvalue,
        'delta_chisqr': delta_chisqr}
    chisqr_df = pd.DataFrame(chisq_dict, index=[0])
    
    
    # ---PLOTTING--- #
    if plot:
    
        ylim = (-1.8, 1.9)

        text_pos1 = (-4.1, 1.55)
        text_pos2 = (-4.1, 1.3)
        text_pos3 = (-4.1, 1.1)
        text_hozalign = 'left'
        
        plt.figure(figsize=[6,5])
        plt.scatter(comb_other_df['feh'], comb_other_df['bestfit_abund_xfe'], color=other_color, marker=other_marker, 
                        edgecolor='black', zorder=10, linewidths=1, label=other_label)
        plt.scatter(comb_halo_df['feh'], comb_halo_df['bestfit_abund_xfe'], c="lightgrey", marker='.', zorder=1, label='Halo Background')

        # Plotting the other fit
        xline = np.linspace(xlim[0], xlim[1], 100)
        yline = fit_other(xline)
        plt.plot(xline, yline, color=other_color, linestyle='--', linewidth=2, label=f'{other_label} Fit')
        plt.fill_between(xline, 
                                yline - std_other,
                                yline + std_other,
                                color=other_color,
                                alpha=0.1)
        
        # Plotting the halo fit
        xline = np.linspace(xlim[0], xlim[1], 100)
        yline = fit_halo(xline)
        plt.plot(xline, yline, color='orangered', linestyle='--', linewidth=2, label='Halo Fit')
        plt.fill_between(xline, 
                                yline - std_halo,
                                yline + std_halo,
                                color='orangered',
                                alpha=0.1)
        
        # Making the text
        plt.text(text_pos1[0], text_pos1[1], other_label, fontsize=12, fontweight='bold', horizontalalignment=text_hozalign)
        plt.text(text_pos2[0], text_pos2[1], f"Chisqr: halo = {round(other_halofit_chisqr, 2)}, {other_label} = {round(other_origfit_chisqr, 2)}", 
                fontsize=12, horizontalalignment=text_hozalign)
        plt.text(text_pos3[0], text_pos3[1], f"Delta chisqr = {round(delta_chisqr, 2)}", fontsize=12, horizontalalignment=text_hozalign)
        
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('[Fe/H]')
        plt.ylabel('[Sc/Mg]')
        # plt.show()
        
    if save_path is not None:
        chisqr_df.to_csv(save_path + f'deltachisqr_niter{n_iter}.csv', index=False)
        
        if plot:
            plt.savefig(save_path + f'deltachisqr_plot_niter{n_iter}.png', dpi=300, transparent=False, facecolor="white", bbox_inches='tight')
        
    return chisqr_df