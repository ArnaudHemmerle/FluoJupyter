from . import Functions
from lib.extraction.common import PyNexus as PN
import lib.frontend as FE

import matplotlib.pyplot as plt
import time
from matplotlib.pyplot import cm
import matplotlib.colors as mplcolors
from matplotlib.ticker import FormatStrFormatter

import numpy as np
# Avoid warning from explosive exponential
np.seterr(all='ignore')
from numpy import linalg
from math import isclose
from lmfit import Minimizer, Parameters, fit_report, conf_interval, printfuncs

import os
import shutil
import csv

from IPython.display import clear_output


"""
Here are defined the functions for analysis.

Description of the different dictionaries of parameters:
- dparams_fit contains the fitting parameters common to the whole spectrum, such as fano, fG, etc ...
- dparams_0 contains the initial guess for the LM fit
- dparams_lm contains the current parameter set during the LM fit
- dparams_list contains lists of results from the fits (one list per fit parameter)
"""


def Fit_spectrums(expt, is_save=True):
    """
    Fit procedure. Will fit each spectrum in expt.spectrums
    If is_save=True, results are saved in FitResults.csv and FitParameters.csv
    """

    #####################################################
    ################   PREPARE FIT   ####################
    #####################################################

    eV = expt.eV
    groups = expt.groups
    dparams_fit = {  'sl':expt.sl,
                     'ct':expt.ct,
                     'sfa0':expt.sfa0,
                     'tfb0':expt.tfb0,
                     'twc0':expt.twc0,
                     'noise':expt.noise,
                     'fano':expt.fano,
                     'epsilon':expt.epsilon,
                     'fG':expt.fG,
                     'fA':expt.fA,
                     'fB':expt.fB,
                     'gammaA':expt.gammaA,
                     'gammaB':expt.gammaB
                     }
    expt.dparams_fit=dparams_fit

    # Dictionnary with a list for each parameter updated at each fit
    dparams_list = {'sl_list', 'ct_list',
                    'sfa0_list', 'tfb0_list', 'twc0_list',
                    'noise_list', 'fano_list', 'epsilon_list',
                    'fG_list', 'fA_list', 'fB_list', 'gammaA_list', 'gammaB_list'
                    }

    # Init all the lists
    dparams_list = dict.fromkeys(dparams_list, np.array([]))

    for group in groups:
        group.area_list = np.array([])
        for peak in group.peaks:
            peak.position_list = np.array([])

    #####################################################
    ##############   PREPARE SAVE   #####################
    #####################################################
    if is_save:
        # Save the results (parameters)       
        # Extract the 0D stamps and data
        nexus = PN.PyNexusFile(expt.path)
        stamps0D, data0D = nexus.extractData('0D')
        nexus.close()
        
        # Prepare the header of the csv file
        header = np.array([])
        
        # Data stamps
        for i in range(len(stamps0D)):
             if (stamps0D[i][1]==None):
                    header =np.append(header, '#'+stamps0D[i][0])
             else:
                    header =np.append(header, '#'+stamps0D[i][1])
        
        # Stamps from the fit            
        for group in groups:
            header = np.append(header, '#'+group.name+'.area')
                
            for peak in group.peaks:
                header = np.append(header,'#'+group.elem_name+'_'+peak.name+'.position')

        for name in dparams_list:
            header = np.append(header, '#'+name[:-5])                    
                    
        with open(expt.working_dir+expt.id+'/FitResults.csv', "w", newline='') as f:
            writer = csv.writer(f,delimiter=expt.delimiter)
            writer.writerow(header)

        # Save the results (fits)
        # Prepare the header of the csv file
        header = np.array(['#sensorsRelTimestamps', '#eV', '#data', '#fit'])
        with open(expt.working_dir+expt.id+'/FitSpectrums.csv', "w", newline='') as f:
            writer = csv.writer(f,delimiter=expt.delimiter)       
            writer.writerow(header)
            
    count=0
    for spectrum in expt.spectrums:

        #####################################################
        #####################   FIT   #######################
        #####################################################

        # Allow or not transmission of current fit params as init params for the next fit 
        is_transmitted = expt.is_transmitted
        if is_transmitted:
            # Initial guess for peak params
            if count==0:
                # First loop : initial guess is given by the expert fit parameters
                dparams_0 = dparams_fit.copy()
            else:
                # For the next loops, initial guess is given by the results of the previous fit
                dparams_0 = {}
                for name in dparams_list:
                    dparams_0[name[:-5]] = dparams_list[name][-1]
        else:
            # Initial guess is always given by the expert fit parameters
            dparams_0 = dparams_fit.copy()

        # Least squares fit
        # Find the least squares solution to the equation ax=b, used as initial guess for the LM fit.
        # p contains the best fit for the area of each group (group.area)
        # Results with the subscript ls
        a = []
        for group in groups:
            spectrum_group = 0.
            for peak in group.peaks:
                    position = peak.position_init
                    intensity_rel = peak.intensity_rel
                    if group.elem_name == 'Compton':
                        spectrum_group += Functions.Fcn_compton_peak(position,intensity_rel,eV,dparams_0)
                    else:
                        spectrum_group += Functions.Fcn_peak(position,intensity_rel,eV,dparams_0)[0]
            a.append(spectrum_group)
        a = np.transpose(a)
        b = spectrum

        area_ls, residues, rank, sv = linalg.lstsq(a,b,1.e-10)

        # Store the group.area of each group
        i=0
        for group in groups:
            group.area_ls = area_ls[i]
            i+=1

        ###################################
        # LMFIT
        dparams_lm = Parameters()

        for group in groups:
            dparams_lm.add('area_'+group.name, value=group.area_ls, vary=True, min = 0.)
            for peak in group.peaks:
                dparams_lm.add('intensity_rel_'+group.name+'_'+peak.name, value=peak.intensity_rel,
                                       vary=False)

                # Check whether the position of the peak should be fitted
                if peak.is_fitpos:
                    dparams_lm.add('pos_'+group.name+'_'+peak.name, value=peak.position_init,
                                   vary = True, min = peak.position_init-100, max = peak.position_init+100)
                else:
                    dparams_lm.add('pos_'+group.name+'_'+peak.name, value=peak.position_init,
                                   vary = False)


        dparams_lm.add('sl', value=dparams_0['sl'])
        dparams_lm.add('ct', value=dparams_0['ct'])
        dparams_lm.add('sfa0', value=dparams_0['sfa0'], min = 0., max = 5.)
        dparams_lm.add('tfb0', value=dparams_0['tfb0'], min = 0., max = 1.)
        dparams_lm.add('twc0', value=dparams_0['twc0'], min = 0., max = 1.)
        dparams_lm.add('noise', value=dparams_0['noise'], min = 0.05, max = 0.2)
        dparams_lm.add('fG', value=dparams_0['fG'], min = 0., max = 2.)
        dparams_lm.add('fA', value=dparams_0['fA'], min = 0., max = 1.)
        dparams_lm.add('fB', value=dparams_0['fB'], min = 0., max = 1.)
        dparams_lm.add('gammaA', value=dparams_0['gammaA'], min = 0., max = 10.)
        dparams_lm.add('gammaB', value=dparams_0['gammaB'], min = 0., max = 10.)
        dparams_lm.add('fano', value=dparams_0['fano'], vary=False)
        dparams_lm.add('epsilon', value=dparams_0['epsilon'], vary=False)

        # Check in list_isfit which peak params should be fitted
        # By default vary = True in dparams_lm.add
        for name in expt.dparams_fit:
            dparams_lm[name].vary = False
            dparams_lm[name].vary = name in expt.list_isfit

        expt.is_fitstuck = False   
        def iter_cb(params, nb_iter, resid, *args, **kws):

            # Stop the current fit if it is stuck or if the spectrum is empty
            if nb_iter > expt.fitstuck_limit:
                expt.is_fitstuck = True

            if np.sum(spectrum)<10.:
                expt.is_fitstuck = True

            return expt.is_fitstuck

        # Do the fit, here with leastsq model
        minner = Minimizer(Fcn2min, dparams_lm, fcn_args=(groups, eV, spectrum), iter_cb=iter_cb, xtol = 1e-6, ftol = 1e-6)

        result = minner.minimize(method = 'leastsq')

        # Calculate final result with the residuals
        #final = spectrum + result.residual

        # Extract the results of the fit and put them in lists

        if expt.is_fitstuck:

            # If the fit was stuck we put NaN
            for group in groups:
                group.area_list = np.append(group.area_list, np.nan)

                for peak in group.peaks:
                    peak.position_list = np.append(peak.position_list, np.nan)

            for name in dparams_list:
                dparams_list[name] =  np.append(dparams_list[name], np.nan)

        else:
            for group in groups:
                group.area_list = np.append(group.area_list, result.params['area_'+group.name].value)

                for peak in group.peaks:
                    peak.position_list = np.append(peak.position_list, result.params['pos_'+group.name+'_'+peak.name].value)

            for name in dparams_list:
                dparams_list[name] =  np.append(dparams_list[name], result.params[name[:-5]].value)

        # Update the dparams_list in expt
        expt.dparams_list = dparams_list
                       
        #####################################################
        ###############   PLOT CURRENT FIT  #################
        #####################################################

        # Plot the spectrum and the fit, updated continuously

        # Dictionnary with the results of the last fit iteration
        dparams = {}
        for name in dparams_list:
            dparams[name[:-5]] = dparams_list[name][-1]
        spectrum_fit, gau_tot, she_tot, tail_tot, baseline, compton = Functions.Fcn_spectrum(dparams, groups, eV)

        clear_output(wait=True) # This line sets the refreshing
        fig = plt.figure(figsize=(15,10))
        fig.suptitle('Fit of spectrum %g/%g'%(count,(len(expt.spectrums)-1)), fontsize=14)
        fig.subplots_adjust(top=0.95)
        ax1 = fig.add_subplot(211)        
        ax1.set(xlabel = 'E (eV)', ylabel = 'counts')
        
        """
        # Uncomment to have all peaks displayed
        colors = iter(['#006BA4', '#FF800E', '#ABABAB', '#595959', 'k', '#C85200', 'b', '#A2C8EC', '#FFBC79']*20)
        linestyles = iter(['--', '-.', '-', ':']*40)  
        for group in groups: 
            for peak in group.peaks:
                position = peak.position_list[-1]
                ax1.axvline(x = position,  color = next(colors) , label = group.name+' '+peak.name)
        """
        
        ax1.plot(eV, spectrum, 'k.')
        ax1.plot(eV, spectrum_fit, 'r-', label = 'Fit', linewidth = 2)
        ax1.legend()
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = fig.add_subplot(212)        
        
        """
        # Uncomment to have all peaks displayed
        colors = iter(['#006BA4', '#FF800E', '#ABABAB', '#595959', 'k', '#C85200', 'b', '#A2C8EC', '#FFBC79']*20)
        linestyles = iter(['--', '-.', '-', ':']*40) 
        for group in groups: 
            for peak in group.peaks:
                position = peak.position_list[-1]
                ax1.axvline(x = position,  color = next(colors) , label = group.name+' '+peak.name)
        """

        ax2.plot(eV, spectrum, 'k.')
        ax2.plot(eV, spectrum_fit, 'r-', linewidth = 2)
        
        if expt.is_show_subfunctions:
            ax2.plot(eV,gau_tot, 'm--', label = 'Gaussian')
            ax2.plot(eV,she_tot, 'g-',label = 'Step')
            ax2.plot(eV,tail_tot, 'b-', label = 'Low energy tail')
            ax2.plot(eV,baseline, 'k-',label = 'Continuum')
            ax2.plot(eV,compton, color = 'grey', linestyle = '-',label = 'Compton')
            ax2.legend(loc = 0)
        ax2.set_ylim(1,1e6)
        ax2.set_yscale('log')
        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        plt.subplots_adjust(hspace=.0)
        plt.show()

        
        
        #####################################################
        #####################   SAVE   ######################
        #####################################################

        # Save the results from the fit
        if is_save:
            
            # Saving FitResults
            # Array to be written
            tbw = np.array([], dtype='float')

            # Put the data0D
            for i in range(len(data0D)):
                 tbw = np.append(tbw, data0D[i][expt.first_spectrum+count])

            # Put the results from the fit
            for group in groups:
                tbw = np.append(tbw,group.area_list[-1])
                for peak in group.peaks:
                    tbw = np.append(tbw,peak.position_list[-1])
            for name in dparams_list:
                tbw = np.append(tbw,dparams_list[name][-1])    

            with open(expt.working_dir+expt.id+'/FitResults.csv', 'a+', newline='') as f:
                writer = csv.writer(f,delimiter=expt.delimiter)
                writer.writerow(tbw)

            # Saving FitSpectrums
            with open(expt.working_dir+expt.id+'/FitSpectrums.csv', 'a+', newline='') as f:
                for i in range(len(eV)):
                    writer = csv.writer(f,delimiter=expt.delimiter)
                    tbw = [expt.sensorsRelTimestamps[count],
                           np.round(eV[i],2),
                           np.round(spectrum[i],2),
                           np.round(spectrum_fit[i],2)]
                    writer.writerow(tbw)                

        count+=1
    #####################################################
    ##################   PLOT PARAMS  ###################
    #####################################################

    # At the end of the fit, plot the evolution of each param as a function of the scan
    print('####################################################')
    print('Fits are done. Results shown below.')
    print('####################################################')
    print('')

    # Results after fits
    expt.dparams_list = dparams_list

    FE.Treatment.Plot_fit_results(expt, spectrum_index=None, dparams_list=dparams_list, is_save=is_save)

    print('#####################################################')
    print('Results are saved in:\n%s'%(expt.working_dir+expt.id+'/FitResults.csv'))

        
def Fcn2min(dparams, groups, eV, data):
    """
    Define objective function: returns the array to be minimized in the lmfit.
    """
    for group in groups:
        group.area = dparams['area_'+group.name]

        for peak in group.peaks:
            peak.position = float(dparams['pos_'+group.name+'_'+peak.name])

    model = Functions.Fcn_spectrum(dparams, groups, eV)[0]

    return model - data
