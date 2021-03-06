from . import Utils
from . import Action

from lib.extraction.common import PyNexus as PN
from lib.extraction import XRF as XRF
from lib.fit import Fit as Fit
from lib.fit import Functions as Functions

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import time
from matplotlib.pyplot import cm
import matplotlib.colors as mplcolors
from matplotlib.ticker import FormatStrFormatter

import numpy as np
from math import isclose

import os
import shutil
import csv

import base64
from IPython.display import clear_output, Javascript, display, HTML
import subprocess

try:
    import ipysheet
except:
    print('Careful: the module ipysheet is not installed!')

try:
    import xraylib
except:
    print('Careful: the module xraylib is not installed!')


"""Frontend library for all the widgets concerning the Treatments in the notebook."""

# Styling options for widgets
style = {'description_width': 'initial'}
tiny_layout = widgets.Layout(width='150px', height='40px')
short_layout = widgets.Layout(width='200px', height='40px')
medium_layout = widgets.Layout(width='250px', height='40px')
large_layout = widgets.Layout(width='300px', height='40px')


def Choose(expt):
    '''
    Choose the Treatment to be applied to the selected scan.

    Parameters
    ----------
    expt : object
        object from the class Experiment
    '''
       
    def on_button_set_params_extract_clicked(b):
        """Set the parameters for extraction, and extract the scan."""

        # Reset the buttons
        expt.is_extract_done = False
        expt.is_fit_params_done = False
        expt.is_fit_done = False

        # Set the parameters
        Set_params_extract(expt)
    
    def on_button_set_params_fit_clicked(b):
        """Set the parameters for the fits."""

        # Reset the buttons
        expt.is_fit_params_done = False
        expt.is_fit_done = False

        # Set the parameters
        Set_params_fit(expt)
        
    def on_button_plot_peaks_clicked(b):
        """Plot the peaks."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        # Reput the sheet set peaks
        Set_peaks(expt)

        # Extract the info from the sheet
        Extract_groups(expt)

        # Display the peaks on the selected spectrum or on the sum
        if expt.is_peaks_on_sum:
            Display_peaks(expt)
        else:
            w = widgets.interact(Display_peaks,
                                 expt = widgets.fixed(expt),
                                 spectrum_index=widgets.IntText(value=0, step=1, description='Spectrum:'))

    def on_button_save_params_fit_clicked(b):
        """Save the current fit params as default ones."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        # Copy the current params as the defaut file
        shutil.copy(expt.working_dir+expt.id+'/Parameters_fit.csv','parameters/fit/Default_Parameters_fit.csv')

        print("Current set of fit parameters saved as default.")
    
    def on_button_load_params_fit_clicked(b):
        """Load a set of fit params as the current one."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        list_params_files = [file for file in sorted(os.listdir('parameters/fit/'))][::-1]

        w_select_files = widgets.Dropdown(options=list_params_files)
    
        def on_button_import_clicked(b):
            "Make the copy."

            # Copy the selected params as the current params file
            shutil.copy('parameters/fit/'+w_select_files.value, expt.working_dir+expt.id+'/Parameters_fit.csv')

            print(str(w_select_files.value)+" imported as current set of fit parameters.")

        button_import = widgets.Button(description="OK",layout=widgets.Layout(width='100px'))
        button_import.on_click(on_button_import_clicked)

        display(widgets.HBox([w_select_files, button_import]))

    def on_button_save_params_peaks_clicked(b):
        """Save the current peaks as default ones."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        # Copy the current peaks as the defaut file
        shutil.copy(expt.working_dir+expt.id+'/Parameters_peaks.csv','parameters/peaks/Default_Parameters_peaks.csv')

        print("Current set of peaks saved as the default one.")

    def on_button_load_params_peaks_clicked(b):
        """Load the default peaks as current ones."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        list_params_files = [file for file in sorted(os.listdir('parameters/peaks/'))][::-1]

        w_select_files = widgets.Dropdown(options=list_params_files)
    
        def on_button_import_clicked(b):
            "Make the copy."

            # Copy the selected params as the current params file
            shutil.copy('parameters/peaks/'+w_select_files.value, expt.working_dir+expt.id+'/Parameters_peaks.csv')

            print(str(w_select_files.value)+" imported as current set of peaks parameters.")

        button_import = widgets.Button(description="OK",layout=widgets.Layout(width='100px'))
        button_import.on_click(on_button_import_clicked)

        display(widgets.HBox([w_select_files, button_import]))
        
        
    def on_button_export_clicked(b):
        """Export the notebook to PDF."""
        
        print('Export in progress...')
        
        export_done = Action.Export_nb_to_pdf(expt.notebook_name)
        
        if export_done:
            print('Notebook exported to %s.pdf'%expt.notebook_name.split('.')[0])
        else:
            print("There was something wrong with the export to pdf.")
            print("Did you rename the Notebook? If yes:")
            print("1) Change the value of expt.notebook_name in the first cell (top of the Notebook).")
            print("2) Re-execute the first cell.")
            print("3) Try to export the pdf again in the last cell (bottom of the Notebook).")
    
    def on_button_next_clicked(b):
        #clear_output(wait=False)
        
        Utils.Delete_current_cell()
        
        Utils.Create_cell(code='FE.Action.Choose(expt)',
                    position ='at_bottom', celltype='code', is_print=False)        
        
    def on_button_markdown_clicked(b):
        """
        Insert a markdown cell below the current cell.
        """ 
        Utils.Delete_current_cell()
        
        Utils.Create_cell(code='', position ='below', celltype='markdown', is_print=True, is_execute=False)
    
        Utils.Create_cell(code='FE.Treatment.Choose(expt)', position ='at_bottom', celltype='code', is_print=False)

    def on_button_start_fit_clicked(b):
        """Start the fit."""

        # Clear the plots
        clear_output(wait=True)

        # Do the fit
        Fit.Fit_spectrums(expt)

        expt.is_fit_done = True

        # Reput the boxes
        Choose(expt)

    def on_button_add_plot_clicked(b):
        """Create a new cell with the result to be added to the report."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        Choose_spectrum_to_plot(expt)        

    def on_button_extract_mean_clicked(b):
        """Extract the mean values of the fitted parameters."""

        for name in expt.dparams_list:
            if name[:-5] in expt.list_isfit:
                  print(name[:-5], np.nanmean(expt.dparams_list[name]))

                  if name[:-5] == 'eV0':
                      expt.eV0 = round(np.nanmean(expt.dparams_list[name]),2)             
          
                  if name[:-5] == 'gain':
                      expt.gain = round(np.nanmean(expt.dparams_list[name]),4)          
          
                  if name[:-5] == 'sl':
                      expt.sl = round(np.nanmean(expt.dparams_list[name]),8)

                  if name[:-5] == 'ct':
                      expt.ct = round(np.nanmean(expt.dparams_list[name]),3)

                  if name[:-5] == 'noise':
                      expt.noise = round(np.nanmean(expt.dparams_list[name]),5)

                  if name[:-5] == 'sfa0':
                      expt.sfa0 = round(np.nanmean(expt.dparams_list[name]),8)

                  if name[:-5] == 'tfb0':
                      expt.tfb0 = round(np.nanmean(expt.dparams_list[name]),8)

                  if name[:-5] == 'twc0':
                      expt.twc0 = round(np.nanmean(expt.dparams_list[name]),8)

                  if name[:-5] == 'fG':
                      expt.fG = round(np.nanmean(expt.dparams_list[name]),8)

                  if name[:-5] == 'fA':
                      expt.fA = round(np.nanmean(expt.dparams_list[name]),8)

                  if name[:-5] == 'fB':
                      expt.fB = round(np.nanmean(expt.dparams_list[name]),8)

                  if name[:-5] == 'gammaA':
                      expt.gammaA = round(np.nanmean(expt.dparams_list[name]),8)

                  if name[:-5] == 'gammaB':
                      expt.gammaB = round(np.nanmean(expt.dparams_list[name]),8)

        # Save the updated values
        # Prepare the header of the csv file
        with open(expt.working_dir+expt.id+'/Parameters_fit.csv', "w", newline='') as f:
            writer = csv.writer(f,delimiter=';',dialect='excel')
            header = np.array([
                    'gain',
                    'eV0',
                    'beam_energy',
                    'fitstuck_limit',
                    'min_strength',
                    'list_isfit_str',
                    'sl',
                    'ct',
                    'noise',
                    'sfa0',
                    'tfb0',
                    'twc0',
                    'fG',
                    'fA',
                    'fB',
                    'gammaA',
                    'gammaB',
                    'epsilon',
                    'fano',
                    'is_transmitted',
                    'is_peaks_on_sum',
                    'is_show_peaks',
                    'is_show_zooms',
                    'is_show_subfunctions',
                    'is_ipysheet'
                    ])
            writer.writerow(header)

            writer.writerow([
                    expt.gain,
                    expt.eV0,
                    expt.beam_energy,
                    expt.fitstuck_limit,
                    expt.min_strength,                    
                    expt.list_isfit_str,
                    expt.sl,
                    expt.ct,
                    expt.noise,
                    expt.sfa0,
                    expt.tfb0,
                    expt.twc0,
                    expt.fG,
                    expt.fA,
                    expt.fB,
                    expt.gammaA,
                    expt.gammaB,
                    expt.epsilon,
                    expt.fano,
                    expt.is_transmitted,
                    expt.is_peaks_on_sum,
                    expt.is_show_peaks,
                    expt.is_show_zooms,
                    expt.is_show_subfunctions,
                    expt.is_ipysheet
                    ])       
        
    # Create the buttons    
  
    button_export = widgets.Button(description="Export to pdf", layout=widgets.Layout(width='180px'))
    button_export.on_click(on_button_export_clicked)

    button_set_params_extract = widgets.Button(description="Extract the scan",layout=widgets.Layout(width='200px'))
    button_set_params_extract.on_click(on_button_set_params_extract_clicked)

    button_set_params_fit = widgets.Button(description="Set parameters",layout=widgets.Layout(width='200px'))
    button_set_params_fit.on_click(on_button_set_params_fit_clicked)

    button_save_params_fit = widgets.Button(description="Save fit params as default",layout=widgets.Layout(width='250px'))
    button_save_params_fit.on_click(on_button_save_params_fit_clicked)

    button_load_params_fit = widgets.Button(description="Load fit params",layout=widgets.Layout(width='200px'))
    button_load_params_fit.on_click(on_button_load_params_fit_clicked)
        
    button_next = widgets.Button(description="Analyze a new scan")
    button_next.on_click(on_button_next_clicked)
    
    button_markdown = widgets.Button(description="Insert comment")
    button_markdown.on_click(on_button_markdown_clicked)
    
    button_plot_peaks = widgets.Button(description="Set peaks",layout=widgets.Layout(width='200px'))
    button_plot_peaks.on_click(on_button_plot_peaks_clicked)
    
    button_load_params_peaks = widgets.Button(description="Load peaks params",layout=widgets.Layout(width='250px'))
    button_load_params_peaks.on_click(on_button_load_params_peaks_clicked)
    
    button_save_params_peaks = widgets.Button(description="Save peaks params as default",layout=widgets.Layout(width='250px'))
    button_save_params_peaks.on_click(on_button_save_params_peaks_clicked)

    button_start_fit = widgets.Button(description="Start fit",layout=widgets.Layout(width='200px'))
    button_start_fit.on_click(on_button_start_fit_clicked)    

    button_add_plot = widgets.Button(description="Add a plot to report",layout=widgets.Layout(width='200px'))
    button_add_plot.on_click(on_button_add_plot_clicked)

    button_extract_mean = widgets.Button(description="Extract averages",layout=widgets.Layout(width='200px'))
    button_extract_mean.on_click(on_button_extract_mean_clicked)    
    
    # Display the widgets depending on the state
    display(widgets.HBox([button_set_params_extract, button_next, button_markdown, button_export]))
    print(100*"-")
    
    if expt.is_extract_done:
        
        print("Set fit parameters:")
        display(widgets.HBox([button_set_params_fit, button_load_params_fit, button_save_params_fit]))
        print(100*"-")       
        
        if expt.is_fit_params_done:
            
            print("Set peaks parameters:")
            display(widgets.HBox([button_plot_peaks, button_load_params_peaks, button_save_params_peaks]))
            print(100*"-")            

            if expt.is_set_peaks_done:

                print("Fit:")
                if expt.is_fit_done:
                    display(widgets.HBox([button_start_fit, button_add_plot, button_extract_mean]))
                else:
                    display(widgets.HBox([button_start_fit,  button_add_plot]))

                print(100*"-")
        
    
def Set_params_extract(expt):
    '''
    Display the widgets for setting the parameters for extraction, and extract the scan.

    Parameters
    ----------
    expt : object
        object from the class Experiment
    ''' 

    # Clear the output
    clear_output(wait=True)

    # Quick extraction of scan info
    nexus = PN.PyNexusFile(expt.path)

    # Extract number of spectrums taken during the scan
    expt.nb_allspectrums = nexus.get_nbpts()
    print("There are %g spectrums in the scan."%(expt.nb_allspectrums))

    # Extract list of detector elements available
    stamps = nexus.extractStamps()
    fluospectrums_available = []
    for i in range(len(stamps)):
        if (stamps[i][1] != None and "fluospectrum0" in stamps[i][1].lower()):
            fluospectrums_available.append(stamps[i][1].lower()[-1])

    print("List of available elements: ", ["Element %s"%s for s in fluospectrums_available])

    def on_button_extract_clicked(b):
        """Extract and plot the scan."""

        # Update the parameters with current values
        update_params_extract()

        # Clear the plots and reput the widgets
        Set_params_extract(expt)

        print("Extraction of:\n%s"%expt.path)

        print(PN._RED+"Wait for the extraction to finish..."+PN._RESET)

        # Create list of SDD elements from booleans
        # Get the chosen fluospectrums
        expt.fluospectrums_chosen = np.array([expt.is_fluospectrum00,expt.is_fluospectrum01,
                                     expt.is_fluospectrum02,expt.is_fluospectrum03, expt.is_fluospectrum04])
        
        temp_elems = expt.fluospectrums_chosen*[10,11,12,13,14]
        expt.list_elems = [i-10 for i in temp_elems if i>0]
        
        # Extract the XRF over the whole range of channels and non-zero spectrums
        expt.channels0, expt.eVs0, expt.spectrums0, expt.first_non_zero_spectrum, expt.last_non_zero_spectrum = \
        XRF.Extract(nxs_filename = expt.nxs, recording_dir = expt.recording_dir,
                    list_elems = expt.list_elems, logz = True,
                    first_channel = 0, last_channel = 2048,
                    gain = 1., eV0 = 0., 
                    fast = expt.is_fast, show_data_stamps = False, verbose = False)

        print('File empty after spectrum %g.'%expt.last_non_zero_spectrum)
        
        # Subset of channels and spectrums defined by user
        expt.channels = np.arange(expt.first_channel, expt.last_channel+1)
        expt.spectrums = expt.spectrums0[expt.first_spectrum:expt.last_spectrum+1,
                                         expt.first_channel:expt.last_channel+1]
        
        # Subset of timestamps (for later saving of the data)
        expt.nexus = PN.PyNexusFile(expt.path, fast=True)
        stamps, data= expt.nexus.extractData(which='0D')
        # Extract timestamps
        for i in range(len(stamps)):
            if (stamps[i][1]== None and stamps[i][0] in ['sensorsRelTimestamps', 'sensors_rel_timestamps']):
                expt.allsensorsRelTimestamps = data[i]
        expt.sensorsRelTimestamps = expt.allsensorsRelTimestamps[expt.first_spectrum:expt.last_spectrum+1]

        #Plot the whole spectrum range (stopping at the last non-zero spectrum).
        #Used to check which subset the user wants.
        
        fig = plt.figure(figsize=(12,6))
        fig.suptitle(expt.nxs, fontsize=14)
        ax1 = fig.add_subplot(111)
        ax1.set_title('All the spectrums in the file (stopping at the last non-zero spectrum)')
        ax1.set(xlabel = 'spectrum index', ylabel = 'channel')
        ax1.set_xlim(left = -1, right = expt.last_non_zero_spectrum+1)
        ax1.axvline(expt.first_spectrum, linestyle = '--', color = 'y', label = 'Selected spectrum range')
        ax1.axvline(expt.last_spectrum, linestyle = '--', color = 'y')
        im1 = ax1.imshow(expt.spectrums0.transpose(), cmap = 'viridis', aspect = 'auto', norm=mplcolors.LogNorm())
        plt.legend()


        # Plot the whole channel range
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        ax1.set_title('Whole range of channels on the sum of all spectrums')
        ax1.set(xlabel = 'channel', ylabel = 'counts')
        ax1.axvline(expt.first_channel, linestyle = '--', color = 'r', label = 'Selected channel range')
        ax1.axvline(expt.last_channel, linestyle = '--', color = 'r')
        ax1.plot(np.arange(2048), expt.spectrums0.sum(axis = 0), 'k.-')
        ax1.legend()
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = fig.add_subplot(212)
        ax2.set(xlabel = 'channel', ylabel = 'counts')
        ax2.axvline(expt.first_channel, linestyle = '--', color = 'r')
        ax2.axvline(expt.last_channel, linestyle = '--', color = 'r')
        ax2.plot(np.arange(2048), expt.spectrums0.sum(axis = 0), 'k.-')
        ax2.set_yscale('log')
        ax2.set_ylim(bottom = 1)
        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        plt.subplots_adjust(hspace=.0)

        #Plot the selected spectrum range
        fig = plt.figure(figsize=(12,6))
        fig.suptitle('SELECTED RANGES', fontsize=14)
        ax1 = fig.add_subplot(111)
        ax1.set_title('Subset of spectrums [%g:%g]'%(expt.first_spectrum,expt.last_spectrum))
        ax1.set(xlabel = 'spectrum index', ylabel = 'channel')
        im1 = ax1.imshow(expt.spectrums.transpose(), cmap = 'viridis', aspect = 'auto', norm=mplcolors.LogNorm(),
                         interpolation='none',
                         extent=[expt.first_spectrum,expt.last_spectrum,
                                 expt.last_channel,expt.first_channel])

        #Plot the selected channel range
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        ax1.set_title('Subset of channels [%g:%g]'%(expt.first_channel,expt.last_channel))
        ax1.set(xlabel = 'channel', ylabel = 'counts')
        ax1.plot(expt.channels, expt.spectrums[0], 'r-', label = 'Spectrum %g'%expt.first_spectrum)
        ax1.plot(expt.channels, expt.spectrums[-1], 'b-', label = 'Spectrum %g'%expt.last_spectrum)
        ax1.legend()
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = fig.add_subplot(212)
        ax2.set(xlabel = 'channel', ylabel = 'counts')
        ax2.plot(expt.channels, expt.spectrums[0], 'r-')
        ax2.plot(expt.channels, expt.spectrums[-1], 'b-')
        ax2.set_yscale('log')
        ax2.set_ylim(bottom = 1)
        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        plt.subplots_adjust(hspace=.0)
        
        # Give the info that the extraction was done
        expt.is_extract_done = True
        


    def update_params_extract():
        """Update the parameters for extraction with the current values"""

        expt.is_fluospectrum00 = w_is_fluospectrum00.value
        expt.is_fluospectrum01 = w_is_fluospectrum01.value
        expt.is_fluospectrum02 = w_is_fluospectrum02.value
        expt.is_fluospectrum03 = w_is_fluospectrum03.value
        expt.is_fluospectrum04 = w_is_fluospectrum04.value
        expt.first_channel = w_first_channel.value
        expt.last_channel = w_last_channel.value
        expt.first_spectrum = w_first_spectrum.value
        expt.last_spectrum = w_last_spectrum.value
        expt.is_fast = w_is_fast.value        
                    

        # Prepare the header of the csv file
        with open(expt.working_dir+expt.id+'/Parameters_extraction.csv', "w", newline='') as f:
            writer = csv.writer(f,delimiter=';',dialect='excel')
            header = np.array([
                    'is_fluospectrum00',
                    'is_fluospectrum01',
                    'is_fluospectrum02',
                    'is_fluospectrum03',
                    'is_fluospectrum04',
                    'first_channel',
                    'last_channel',
                    'first_spectrum',
                    'last_spectrum',
                    'is_fast'
                    ])
            writer.writerow(header)

            writer.writerow([
                    expt.is_fluospectrum00,
                    expt.is_fluospectrum01,
                    expt.is_fluospectrum02,
                    expt.is_fluospectrum03,
                    expt.is_fluospectrum04,
                    expt.first_channel,
                    expt.last_channel,
                    expt.first_spectrum,
                    expt.last_spectrum,
                    expt.is_fast
                    ])

        
        
    # Load the scan info from file
    with open(expt.working_dir+expt.id+'/Parameters_extraction.csv', "r") as f:
        reader = csv.DictReader(f, delimiter=';',dialect='excel')
        for row in reader:
            is_fluospectrum00 = eval(row['is_fluospectrum00'])
            is_fluospectrum01 = eval(row['is_fluospectrum01'])
            is_fluospectrum02 = eval(row['is_fluospectrum02'])
            is_fluospectrum03 = eval(row['is_fluospectrum03'])
            is_fluospectrum04 = eval(row['is_fluospectrum04'])
            first_channel = int(row['first_channel'])
            last_channel = int(row['last_channel'])
            first_spectrum = int(row['first_spectrum'])
            last_spectrum = int(row['last_spectrum'])
            is_fast = eval(row['is_fast'])

    def on_button_continue_clicked(b):
        """Call back the main panel"""
       
        # Clear the plots and reput the widgets
        clear_output(wait=True)
        Choose(expt)

    
    w_is_fluospectrum00 = widgets.Checkbox(
        value=is_fluospectrum00,
        style=style,
        description='Element 0')

    w_is_fluospectrum01 = widgets.Checkbox(
        value=is_fluospectrum01,
        style=style,
        description='Element 1')

    w_is_fluospectrum02 = widgets.Checkbox(
        value=is_fluospectrum02,
        style=style,
        description='Element 2')

    w_is_fluospectrum03 = widgets.Checkbox(
        value=is_fluospectrum03,
        style=style,
        description='Element 3')

    w_is_fluospectrum04 = widgets.Checkbox(
        value=is_fluospectrum04,
        style=style,
        description='Element 4')

    w_first_channel = widgets.BoundedIntText(
        value=first_channel,
        min=0,
        max=2048,
        step=1,
        description='First channel',
        layout=widgets.Layout(width='200px'),
        style=style)

    w_last_channel = widgets.BoundedIntText(
        value=last_channel,
        min=0,
        max=2048,
        step=1,
        description='Last channel',
        layout=widgets.Layout(width='200px'),
        style=style)

    w_first_spectrum = widgets.IntText(
        value=first_spectrum,
        step=1,
        description='First spectrum',
        layout=widgets.Layout(width='200px'),
        style=style)

    w_last_spectrum = widgets.IntText(
        value=last_spectrum,
        step=1,
        description='Last spectrum',
        layout=widgets.Layout(width='200px'),
        style=style)

    w_is_fast = widgets.Checkbox(
        value=is_fast,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='Fast extract')    
 

    button_extract = widgets.Button(description="Extract",layout=widgets.Layout(width='200px'))
    button_extract.on_click(on_button_extract_clicked)

    button_continue = widgets.Button(description="Continue",layout=widgets.Layout(width='200px'))
    button_continue.on_click(on_button_continue_clicked)    
    
    display(widgets.HBox([w_first_channel, w_last_channel, w_first_spectrum, w_last_spectrum]))
        
    display(widgets.HBox([w_is_fluospectrum00, w_is_fluospectrum01,w_is_fluospectrum02,
                                   w_is_fluospectrum03,w_is_fluospectrum04, w_is_fast]))

    display(widgets.HBox([button_extract, button_continue]))
    

    
def Set_params_fit(expt):
    '''
    Display the widgets for setting the parameters for fit.

    Parameters
    ----------
    expt : object
        object from the class Experiment
    ''' 

    # Clear the output
    clear_output(wait=True)


    def on_button_validate_clicked(b):
        """Validate the parameters."""

        # Update the parameters with current values
        update_params_fit()

        # Give the info that the extraction was done
        expt.is_fit_params_done = True
        
        clear_output(wait=True)
        Choose(expt)        
        
        
    def update_params_fit():
        """Update the parameters for fits with the current values"""
                     
        expt.gain = w_gain.value
        expt.eV0 = w_eV0.value
        expt.beam_energy = w_beam_energy.value
        expt.fitstuck_limit = w_fitstuck_limit.value
        expt.min_strength = w_min_strength.value
        expt.sl = w_sl.value
        expt.ct = w_ct.value
        expt.noise = w_noise.value
        expt.sfa0 = w_sfa0.value
        expt.tfb0 = w_tfb0.value
        expt.twc0 = w_twc0.value
        expt.fG = w_fG.value
        expt.fA = w_fA.value
        expt.fB = w_fB.value
        expt.gammaA = w_gammaA.value
        expt.gammaB = w_gammaB.value
        expt.epsilon = w_epsilon.value
        expt.fano = w_fano.value
        expt.is_transmitted = w_is_transmitted.value
        expt.is_peaks_on_sum = w_is_peaks_on_sum.value
        expt.is_show_peaks = w_is_show_peaks.value
        expt.is_show_zooms = w_is_show_zooms.value
        expt.is_ipysheet = w_is_ipysheet.value   
        
        try:
            expt.is_show_subfunctions = expt.is_show_subfunctions
        except:
            expt.is_show_subfunctions = True
       

        # Particular case of list_isfit, going from str to array
        list_isfit = ['gain'*w_is_gain.value, 'eV0'*w_is_eV0.value,
                      'sl'*w_is_sl.value, 'ct'*w_is_ct.value, 'noise'*w_is_noise.value,
                      'sfa0'*w_is_sfa0.value, 'tfb0'*w_is_tfb0.value,
                      'twc0'*w_is_twc0.value, 'fG'*w_is_fG.value,
                      'fA'*w_is_fA.value, 'fB'*w_is_fB.value, 'gammaA'*w_is_gammaA.value, 'gammaB'*w_is_gammaB.value,
                      'epsilon'*False, 'fano'*False]

        while("" in list_isfit) :
            list_isfit.remove("")

        expt.list_isfit = list_isfit
        expt.list_isfit_str = ','.join(list_isfit)

       
        # Prepare the header of the csv file
        with open(expt.working_dir+expt.id+'/Parameters_fit.csv', "w", newline='') as f:
            writer = csv.writer(f,delimiter=';',dialect='excel')
            header = np.array([
                    'gain',
                    'eV0',
                    'beam_energy',
                    'fitstuck_limit',
                    'min_strength',
                    'list_isfit_str',
                    'sl',
                    'ct',
                    'noise',
                    'sfa0',
                    'tfb0',
                    'twc0',
                    'fG',
                    'fA',
                    'fB',
                    'gammaA',
                    'gammaB',
                    'epsilon',
                    'fano',
                    'is_transmitted',
                    'is_peaks_on_sum',
                    'is_show_peaks',
                    'is_show_zooms',
                    'is_show_subfunctions',
                    'is_ipysheet'
                    ])
            writer.writerow(header)

            writer.writerow([
                    expt.gain,
                    expt.eV0,
                    expt.beam_energy,
                    expt.fitstuck_limit,
                    expt.min_strength,                    
                    expt.list_isfit_str,
                    expt.sl,
                    expt.ct,
                    expt.noise,
                    expt.sfa0,
                    expt.tfb0,
                    expt.twc0,
                    expt.fG,
                    expt.fA,
                    expt.fB,
                    expt.gammaA,
                    expt.gammaB,
                    expt.epsilon,
                    expt.fano,
                    expt.is_transmitted,
                    expt.is_peaks_on_sum,
                    expt.is_show_peaks,
                    expt.is_show_zooms,
                    expt.is_show_subfunctions,
                    expt.is_ipysheet
                    ])

        
        
    # Load the scan info from file
    with open(expt.working_dir+expt.id+'/Parameters_fit.csv', "r") as f:
        reader = csv.DictReader(f, delimiter=';',dialect='excel')
        for row in reader:
            gain = float(row['gain'].replace(',', '.'))
            eV0 = float(row['eV0'].replace(',', '.'))
            beam_energy = float(row['beam_energy'].replace(',', '.'))
            fitstuck_limit = int(row['fitstuck_limit'])
            min_strength = float(row['min_strength'].replace(',', '.'))
            list_isfit_str = str(row['list_isfit_str'])
            sl = float(row['sl'].replace(',', '.'))
            ct = float(row['ct'].replace(',', '.'))
            noise = float(row['noise'].replace(',', '.'))
            sfa0 = float(row['sfa0'].replace(',', '.'))
            tfb0 = float(row['tfb0'].replace(',', '.'))
            twc0 = float(row['twc0'].replace(',', '.'))
            fG = float(row['fG'].replace(',', '.'))
            fA = float(row['fA'].replace(',', '.'))
            fB = float(row['fB'].replace(',', '.'))
            gammaA = float(row['gammaA'].replace(',', '.'))
            gammaB = float(row['gammaB'].replace(',', '.'))
            epsilon = float(row['epsilon'].replace(',', '.'))
            fano = float(row['fano'].replace(',', '.'))
            is_transmitted = eval(row['is_transmitted'])
            is_peaks_on_sum = eval(row['is_peaks_on_sum'])
            is_show_peaks = eval(row['is_show_peaks'])
            is_show_zooms = eval(row['is_show_zooms'])
            is_show_subfunctions = eval(row['is_show_subfunctions'])
            is_ipysheet = eval(row['is_ipysheet'])

 
    # convert list_isfit_str into a list
    list_isfit = [str(list_isfit_str.split(',')[i]) for i in range(len(list_isfit_str.split(',')))]

    w_gain = widgets.FloatText(
        value=gain,
        description='Gain',
        layout=widgets.Layout(width='120px'),
        style=style)

    w_eV0 = widgets.FloatText(
        value=eV0,
        description='eV0',
        layout=widgets.Layout(width='100px'),
        style=style)
    
    w_beam_energy = widgets.FloatText(
        value=beam_energy,
        description='Energy (eV)',
        layout=widgets.Layout(width='150px'),
        style=style)
    
    w_fitstuck_limit = widgets.IntText(
        value=fitstuck_limit,
        description='Limit iter.',
        layout=widgets.Layout(width='150px'),
        style=style)

    w_min_strength = widgets.FloatText(
        value=min_strength,
        description='Strength min.',
        layout=widgets.Layout(width='180px'),
        style=style)

    # Fit params: boolean
    w_is_gain = widgets.Checkbox(
        value='gain' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='gain')    
    
    w_is_eV0 = widgets.Checkbox(
        value='eV0' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='eV0')

    w_is_gammaA = widgets.Checkbox(
        value='gammaA' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='gammaA')

    w_is_gammaB = widgets.Checkbox(
        value='gammaB' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='gammaB')

    w_is_fA = widgets.Checkbox(
        value='fA' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='fA')

    w_is_fB = widgets.Checkbox(
        value='fB' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='fB')

    w_is_fG = widgets.Checkbox(
        value='fG' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='fG')

    w_is_twc0 = widgets.Checkbox(
        value='twc0' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='twc0')

    w_is_tfb0 = widgets.Checkbox(
        value='tfb0' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='tfb0')

    w_is_sfa0 = widgets.Checkbox(
        value='sfa0' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='sfa0')

    w_is_sl = widgets.Checkbox(
        value='sl' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='sl')

    w_is_ct = widgets.Checkbox(
        value='ct' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='ct')

    w_is_noise = widgets.Checkbox(
        value='noise' in list_isfit,
        style=style,
        layout=widgets.Layout(width='100px'),
        description='noise')

    # Fit params: value

    w_gammaA = widgets.FloatText(
        value=gammaA,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='gammaA')

    w_gammaB = widgets.FloatText(
        value=gammaB,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='gammaB')

    w_fA = widgets.FloatText(
        value=fA,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='fA')

    w_fB = widgets.FloatText(
        value=fB,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='fB')


    w_fG = widgets.FloatText(
        value=fG,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='fG')

    w_twc0 = widgets.FloatText(
        value=twc0,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='twc0')

    w_tfb0 = widgets.FloatText(
        value=tfb0,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='tfb0')

    w_sfa0 = widgets.FloatText(
        value=sfa0,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='sfa0')

    w_sl = widgets.FloatText(
        value=sl,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='sl')

    w_ct = widgets.FloatText(
        value=ct,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='ct')

    w_noise = widgets.FloatText(
        value=noise,
        style=style,
        layout=widgets.Layout(width='200px'),
        description='noise')

    w_epsilon = widgets.FloatText(
        value=epsilon,
        style=style,
        layout=widgets.Layout(width='120px'),
        description='epsilon')

    w_fano = widgets.FloatText(
        value=fano,
        style=style,
        layout=widgets.Layout(width='120px'),
        description='fano')

    w_is_transmitted = widgets.Checkbox(
        value=is_transmitted,
        style=style,
        layout=widgets.Layout(width='150px'),
        description='Transmit fit params')

    w_is_peaks_on_sum = widgets.Checkbox(
        value=is_peaks_on_sum,
        style=style,
        layout=widgets.Layout(width='150px'),
        description='Set peaks on sum')

    w_is_show_peaks = widgets.Checkbox(
        value=is_show_peaks,
        layout=widgets.Layout(width='120px'),
        style=style,
        description='Show peaks?')
    
    w_is_show_zooms = widgets.Checkbox(
        value=is_show_zooms,
        layout=widgets.Layout(width='120px'),
        style=style,
        description='Show zooms?')    
    
    w_is_ipysheet = widgets.Checkbox(
        value=is_ipysheet,
        style=style,
        layout=widgets.Layout(width='150px'),
        description='Use ipysheet')    

    button_validate = widgets.Button(description="Validate",layout=widgets.Layout(width='200px'))
    button_validate.on_click(on_button_validate_clicked)
    
    print("-"*100)
    # Fit params
    print("Params for conversion to eVs: eVs = gain*channels + eV0.")
    print("Params for background. sl: slope, ct: constant.")
    print("Params for peak. noise: width, tfb0: tail fraction, twc0: tail width, sfa0: shelf fraction.")
    print("Params for Compton. fG: width broadening factor, fA/fB: tail fraction at low/high energies, \ngammaA/gammaB: slope at low/high energies.")
    display(widgets.HBox([w_is_gain, w_is_eV0, w_is_sl, w_is_ct]))
    display(widgets.HBox([w_is_noise, w_is_sfa0, w_is_tfb0, w_is_twc0]))
    display(widgets.HBox([w_is_fG, w_is_fA, w_is_fB, w_is_gammaA,w_is_gammaB]))

    display(widgets.HBox([w_sl, w_ct, w_noise]))
    display(widgets.HBox([w_sfa0, w_tfb0, w_twc0]))
    display(widgets.HBox([w_fA, w_fB, w_fG]))
    display(widgets.HBox([w_gammaA, w_gammaB]))

    print("-"*100)
    display(widgets.HBox([w_gain, w_eV0, w_beam_energy,  w_epsilon, w_fano]))
    display(widgets.HBox([w_fitstuck_limit, w_min_strength]))
    display(widgets.HBox([w_is_ipysheet,
                          w_is_transmitted, w_is_peaks_on_sum, w_is_show_peaks, w_is_show_zooms]))

    
    display(widgets.HBox([button_validate]))


def Set_peaks(expt):
    '''
    1) Check if the csv file Peaks.csv exists, if not copy DefaultPeaks.csv in the expt folder
    2) If ipysheet is activated, display the interactive sheet. If not, extract the peaks from Peaks.csv
    3) Save the peaks in Peaks.csv
    Update expt.arr_peaks, with the info on the peaks later used to create the Group and Peak objects.

    Parameters
    ----------
    expt : object
        object from the class Experiment
    '''

    # Array which will contain the info on peaks
    arr_peaks = np.array([])

    # Check if the csv file already exists, if not copy the DefaultPeaks.csv file
    if not os.path.isfile(expt.working_dir+expt.id+'/Parameters_peaks.csv'):
        shutil.copy('parameters/peaks/Default_Parameters_peaks.csv', expt.working_dir+expt.id+'/Parameters_peaks.csv')

    with open(expt.working_dir+expt.id+'/Parameters_peaks.csv', "r") as f:
        csvreader = csv.reader(f, delimiter=';')
        # First line is the header
        expt.peaks_header = next(csvreader)
        nb_columns = len(expt.peaks_header)
        for row in csvreader:
            arr_peaks = np.append(arr_peaks, row)
    arr_peaks = np.reshape(arr_peaks, (len(arr_peaks)//nb_columns,nb_columns))

    # String to print the peaks
    prt_peaks = '\t'.join([str(cell) for cell in expt.peaks_header])+'\n'
    prt_peaks += '\n'.join(['\t \t'.join([str(cell[0:7]) for cell in row]) for row in arr_peaks if row[0]!=''])+'\n'
    prt_peaks += "Peaks saved in:\n%s"%(expt.working_dir+expt.id+'/Parameters_Peaks.csv')

    expt.arr_peaks = arr_peaks
    expt.prt_peaks = prt_peaks

    # Determine the number of rows to have a fixed number of empty rows
    nb_filled_rows = len([elem for elem in arr_peaks if elem[0]!=''])
    nb_empty_rows = len([elem for elem in arr_peaks if elem[0]==''])
    if nb_empty_rows<15:
        nb_rows = nb_filled_rows+15
    else:
        nb_rows = np.shape(arr_peaks)[0]


    if expt.is_ipysheet:
        sheet = ipysheet.easy.sheet(columns=nb_columns, rows=nb_rows ,column_headers = expt.peaks_header)

        # ipysheet does not work correctly with no entries
        # it is necessary to fill first the cells with something
        nb_rows_to_fill = nb_rows - np.shape(arr_peaks)[0]
        fill = np.reshape(np.array(nb_rows_to_fill*nb_columns*['']),
                          (nb_rows_to_fill, nb_columns))
        arr_peaks = np.vstack((arr_peaks, fill))

        for i in range(nb_columns):
            ipysheet.easy.column(i,  arr_peaks[:,i])

        def on_button_update_clicked(b):
            """
            Update the peaks in the sheet by writing in Parameters_peaks.csv and re-displaying the peaks and panel.
            """

            # Give the info that the set peaks was done
            expt.is_set_peaks_done = True
    
            # Clear the plots and reput the boxes
            clear_output(wait=True)
            Choose(expt)

            # Collect the info from the sheet, store them in arr_peaks, write to Parameters_peaks.csv
            arr_peaks = ipysheet.numpy_loader.to_array(ipysheet.easy.current())

            with open(expt.working_dir+expt.id+'/Parameters_peaks.csv', "w", newline='') as f:
                writer = csv.writer(f,delimiter=';')
                writer.writerow(expt.peaks_header)
                writer.writerows(arr_peaks)

            # Reput the sheet set peaks
            Set_peaks(expt)

            # Extract the info from the sheet
            Extract_groups(expt)

            # Display the peaks on the selected spectrum or on the sum
            if expt.is_peaks_on_sum:
                Display_peaks(expt)
            else:
                w = widgets.interact(Display_peaks,
                                     expt=widgets.fixed(expt),
                                     spectrum_index=widgets.IntText(value=0, step=1, description='Spectrum:'))

            expt.arr_peaks = arr_peaks

        def on_button_add_from_db_clicked(b):
            """
            Add a peak from the database.
            The strength of each peak is calculated via:
            - the line intensity relative to the group of peaks with the same initial level (ex. L1)
            - the jump ratio of the initial level
            - the fluo yield of the initial level
            - the probability to transition to another level
            Selection is made through nested interactive widgets.
            The selected group of peaks is then written in Parameters_peaks.csv.
            """

            # Clear the plots and reput the boxes
            clear_output(wait=True)
            Choose(expt)

            # Create the list of atoms based on the database
            atom_name_list = [str(xraylib.AtomicNumberToSymbol(i)) for i in range(1,99)]

            # Construct an array for extracting line names from xraylib
            line_names = []
            with open('lib/extraction/xraylib_lines.pro', "r") as f:
                csvreader = csv.reader(f)
                for row in csvreader:
                    if row!=[]:
                        if (row[0][0]!=';' and row[0]!='end'):
                            line_names.append(row[0].split(' = ')[0].split('_')[0])
            
            def select_group(atom_chosen):
                """
                Widget to show and select the group of peaks to add (from 'K', 'L', or 'M')
                """

                tbw = []
                
                def print_group(group_chosen):
                    """
                    Print the group of peaks.
                    """
                    print('Peaks to be added:')
                    print('')
                    
                    if group_chosen == 'K':
                        ind_min = -29
                        ind_max = 0
                    if group_chosen == 'L':
                        ind_min = -113
                        ind_max = -29
                    if group_chosen == 'M':
                        ind_min = -219
                        ind_max = -113
                    
                    for i in range(ind_min,ind_max):

                        Z = xraylib.SymbolToAtomicNumber(atom_chosen)
                        try:
                            strength =  xraylib.CS_FluorLine_Kissel(Z, i, expt.beam_energy/1000.)
                            energy = xraylib.LineEnergy(Z, i)*1000.

                            # Put an absolute cut-off on the strength
                            if strength>expt.min_strength:          
                                # Array to be written
                                tbw.append([atom_chosen,
                                            line_names[-i-1],
                                            '{:.1f}'.format(energy),
                                            '{:f}'.format(strength),'no','yes'])
                                print('Line name: %s, energy (eV): %g, strength: %g'%(line_names[-i-1], energy, strength))
                                is_line_to_print = True
                        except:
                            pass
                    

                    def on_button_add_group_clicked(b):
                        """
                        Add the group of peaks to the file "Parameters_peaks.csv"
                        """

                        # Rewrite the previous peaks (without the empty lines)
                        with open(expt.working_dir+expt.id+'/Parameters_peaks.csv', "w", newline='') as f:
                            writer = csv.writer(f,delimiter=';')
                            writer.writerow(expt.peaks_header)
                            writer.writerows([elem for elem in expt.arr_peaks_all if elem[0]!=''])


                        # Add the new lines
                        with open(expt.working_dir+expt.id+'/Parameters_peaks.csv', "a", newline='') as f:
                            writer = csv.writer(f,delimiter=';')
                            writer.writerows(tbw)

                        # Reconstruct expt.arr_peaks with the new lines
                        arr_peaks_all = np.array([])
                        with open(expt.working_dir+expt.id+'/Parameters_peaks.csv', "r") as f:
                            csvreader = csv.reader(f, delimiter=';')
                            # First line is the header
                            expt.peaks_header = next(csvreader)
                            for row in csvreader:
                                arr_peaks_all = np.append(arr_peaks_all, row)

                        arr_peaks_all = np.reshape(arr_peaks_all, (len(arr_peaks_all)//nb_columns,nb_columns))
                        expt.arr_peaks_all = arr_peaks_all

                        print(PN._RED+"Done!"+PN._RESET)
                        print(PN._RED+"Click on Set peaks to check peaks."+PN._RESET)
                        

                    # Button to add the displayed group of peaks
                    button_add_group = widgets.Button(description="Add group of peaks",layout=widgets.Layout(width='300px'))
                    button_add_group.on_click(on_button_add_group_clicked)

                    display(button_add_group)

                w_print_group =  widgets.interact(print_group,
                                group_chosen = widgets.Dropdown(
                                options=['K', 'L', 'M'],
                                description='Select level:',
                                layout=widgets.Layout(width='200px'),
                                style=style)
                                )

            w_select_atom = widgets.interact(select_group,
                                atom_chosen = widgets.Dropdown(
                                options=atom_name_list,
                                value = 'Ar',
                                description='Select atom:',
                                layout=widgets.Layout(width='200px'),
                                style=style)
                                )


        button_update = widgets.Button(description="Validate peaks")
        button_update.on_click(on_button_update_clicked)

        button_add_from_db = widgets.Button(description="Add peaks from database", layout=widgets.Layout(width='300px'))
        button_add_from_db.on_click(on_button_add_from_db_clicked)

        display(widgets.HBox([button_update, button_add_from_db]))        
        display(sheet)

    else:
        print("Peaks imported from %s"%(expt.working_dir+expt.id+'/Parameters_peaks.csv'))
        print('\t'.join([str(cell) for cell in expt.peaks_header]))
        print('\n'.join(['\t \t'.join([str(cell) for cell in row]) for row in arr_peaks if row[0]!='']))

        def on_button_validate_clicked(b):
            """
            Goes back to the panel.
            """

            # Give the info that the set peaks was done
            expt.is_set_peaks_done = True
    
            # Clear the plots and reput the boxes
            clear_output(wait=True)
            Choose(expt)
            
        button_validate = widgets.Button(description="OK",layout=widgets.Layout(width='200px'))
        button_validate.on_click(on_button_validate_clicked)

        display(widgets.HBox([button_validate]))
        
        
class Group:
    """
    Class for group of peaks from a same element and with the same family.
    """
    def __init__(self, name):

        # Name of the group elem_familyName (Au_L1M2, Cl_KL3, ...)
        self.name = name

        # Array for peaks
        self.peaks = []        
        

class Peak:
    """
    Class for fluo peak belongin to a Group.
    """

    def __init__(self, name, position_init, strength, is_fitpos = False):

        # Name after the corresponding transition (KL3, L1M3, X, ...)
        self.name = name

        # Position before fit, as given by the user
        self.position_init = position_init

        # Strength of the line (Fluo production cross section)
        self.strength = strength

        # Set if the position of the line is fitted
        self.is_fitpos = is_fitpos
       
        
def Extract_groups(expt):
    '''
    Create objects Group and Peak from info in arr_peaks

    Parameters
    ----------
    expt : object
        object from the class Experiment
    '''
    # Create objects Group and Peak from info in arr_peaks
    # Peaks are grouped by lines belonging to the same fluo element and with the same family K L or M
    # Each "ElemName+FamilyName" gives a new Group with Group.name = "ElemName_FamilyName"
    # Each "LineName" gives a new Group.Peak with Group.Peak.name = "LineName"
    # "Position (eV)" -> Group.Peak.position_init
    # "Strength" -> Group.Peak.strength
    # "Fit position?" -> Group.Peak.is_fitpos
    # The array expt.Groups contains the list of objects Group

    # Remove the peaks which are not fitted from scan.arr_peaks
    expt.arr_peaks_all = expt.arr_peaks
    expt.arr_peaks = expt.arr_peaks[np.where(expt.arr_peaks[:,5]!='no')]

    # List of groups of peaks (same elem, same family K L or M)
    Groups = []

    ###################################################################
    # Construct the groups and lines
    for i in range(np.shape(expt.arr_peaks)[0]):

        elem_name = expt.arr_peaks[i][0]

        if elem_name != '':

            line_name = expt.arr_peaks[i][1]

            # Determine the family name from the line name
            if line_name[0] in ['K', 'L', 'M']:
                family_name = line_name[0]
            else:
                family_name = line_name

            # Define the name of a group of peaks
            group_name = elem_name+'_'+family_name


            # Check if the group has already been created
            is_new_group = True
            for group in Groups:
                if group_name == group.name:
                    is_new_group = False

            # Create a new group
            if is_new_group:
                newGroup = Group(group_name)
                Groups = np.append(Groups, newGroup)
                Groups[-1].elem_name = elem_name
                Groups[-1].peaks = []

            # Convert yes/no in True/False
            if expt.arr_peaks[i][4] == 'yes':
                is_fitpos = True
            else:
                is_fitpos = False

            # Add the peak to the right group
            for group in Groups:
                if group_name == group.name:
                    newPeak = Peak(
                    name = str(expt.arr_peaks[i][1]),
                    position_init = float(expt.arr_peaks[i][2]),
                    strength = float(expt.arr_peaks[i][3]),
                    is_fitpos = is_fitpos)

                    group.peaks = np.append(group.peaks, newPeak)

    expt.groups = Groups

    ###################################################################
    # Compute the relative intensity (relative to the most intense peak,
    # i.e. intensity_rel = 1 for the most intense line of a given family K L or M)

    for group in expt.groups:
        max_strength = 0.

        # Extract the most intense strength of the group
        for peak in group.peaks:
            if peak.strength>max_strength:
                max_strength = peak.strength

        # Normalize the strengths wit the most intense one
        for peak in group.peaks:
            peak.intensity_rel = peak.strength/max_strength

        
def Display_peaks(expt, spectrum_index=0):
    '''
    Plot the position of each peaks on the given spectrum spectrum_index (or the sum).
    Take spectrum_index, the index of which spectrum you want to use.

    Parameters
    ----------
    expt : object
        object from the class Experiment
    spectrum_index : int, optional
        index of the spectrum to be displayed
    '''
    # Convert channels into eV
    expt.eVs = expt.channels*expt.gain + expt.eV0

    if expt.is_peaks_on_sum:
        # We work on the sum to define the peaks
        spectrum = np.sum(expt.spectrums, axis=0)
    else:
        # We work on the spectrum specified by spectrum_index
        spectrum = expt.spectrums[spectrum_index]

    # Plot the spectrum and each line given by the user
    Plot_spectrum(expt, spectrum_index=spectrum_index)

    if expt.is_ipysheet: print(expt.prt_peaks)


def Plot_spectrum(expt, spectrum_index=0, dparams_list=None):
    '''
    Plot data of a specific spectrum (given by spectrum_index).
    If a dparams_list is given, redo and plot the fit with the given parameters.
    
    Parameters
    ----------
    expt : object
        object from the class Experiment
    spectrum_index : int, optional
        index of the spectrum to be displayed
    dparams_list : array_like
        list of parameters to fit    
    '''
    n = spectrum_index
    groups = expt.groups

    if dparams_list != None:
        # We work on the spectrum specified by spectrum_index
        spectrum = expt.spectrums[n]

    else:
        if expt.is_peaks_on_sum:
            # We work on the sum to define the peaks
            spectrum = np.sum(expt.spectrums, axis=0)
        else:
            # We work on the spectrum specified by spectrum_index
            spectrum = expt.spectrums[n]


    if dparams_list != None:
        sl_list = dparams_list['sl_list']
        ct_list = dparams_list['ct_list']

        # Fit the spectrum with the given parameters
        for group in groups:
            group.area = group.area_list[n]
            for peak in group.peaks:
                peak.position = peak.position_list[n]
        dparams = {}
        for name in dparams_list:
            dparams[name[:-5]] = dparams_list[name][n]
        spectrum_fit, gau_tot, she_tot, tail_tot, baseline, compton, eVs =\
                                         Functions.Fcn_spectrum(dparams, groups, expt.channels)

    else:
        eVs = expt.eVs
        for group in groups:
            for peak in group.peaks:
                peak.position = peak.position_init

    # Plot the whole spectrum
    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(211)
    colors = iter(['#006BA4', '#FF800E', '#ABABAB', '#595959', 'k', '#C85200', 'b', '#A2C8EC', '#FFBC79']*200)
    linestyles = iter(['--', '-.', '-', ':']*400)
    ax1.set(xlabel = 'E (eV)', ylabel = 'counts')


    elem_dict = {}
    for group in expt.groups:

        # To have one color/label per elem
        if group.elem_name in elem_dict.keys():
            color = elem_dict[group.elem_name][0]
            linestyle = elem_dict[group.elem_name][1]
            is_first_group_of_elem = False

        else:
            color = next(colors)
            linestyle = next(linestyles)
            elem_dict.update({group.elem_name:(color,linestyle)})
            is_first_group_of_elem = True

        is_first_peak_of_group = True

        for peak in group.peaks:
            position = peak.position
            if (is_first_peak_of_group and is_first_group_of_elem):
                if expt.is_show_peaks:
                    # Plot the peak only if asked
                    ax1.axvline(x = position,  color = color, linestyle = linestyle, label = group.elem_name)
                    is_first_peak_of_group = False
            else:
                if expt.is_show_peaks:
                    ax1.axvline(x = position,  color = color, linestyle = linestyle, label = '')

    ax1.plot(eVs, spectrum, 'k.')
    if dparams_list != None: ax1.plot(eVs, spectrum_fit, 'r-', linewidth = 2)
    if expt.is_show_peaks:  ax1.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)
    for item in ([ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(14)


    ax2 = fig.add_subplot(212)
    colors = iter(['#006BA4', '#FF800E', '#ABABAB', '#595959', 'k', '#C85200', 'b', '#A2C8EC', '#FFBC79']*200)
    linestyles = iter(['--', '-.', '-', ':']*400)
    ax2.set(xlabel = 'E (eV)', ylabel = 'counts')
    for group in groups:

        color = elem_dict[group.elem_name][0]
        linestyle = elem_dict[group.elem_name][1]

        for peak in group.peaks:
            position = peak.position
            if expt.is_show_peaks:
                # Plot the peak only if asked, and if its strength is > than a min value
                ax2.axvline(x = position,  color = color, linestyle = linestyle)

    ax2.plot(eVs, spectrum, 'k.')
    if dparams_list != None:
        ax2.plot(eVs, spectrum_fit, 'r-', linewidth = 2)
        if expt.is_show_subfunctions:
            ax2.plot(eVs,gau_tot, 'm--', label = 'Gaussian')
            ax2.plot(eVs,she_tot, 'g-',label = 'Step')
            ax2.plot(eVs,tail_tot, 'b-', label = 'Low energy tail')
            ax2.plot(eVs,baseline, 'k-',label = 'Continuum')
            ax2.plot(eVs,compton, color = 'grey', linestyle = '-',label = 'Compton')
            ax2.legend(loc = 0)
    ax2.set_ylim(bottom = 1)
    ax2.set_yscale('log')
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    for item in ([ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(14)

    plt.subplots_adjust(hspace=.0)
    fig.subplots_adjust(top=0.95)
    if (expt.is_peaks_on_sum and dparams_list==None):
        fig.suptitle(expt.nxs+': Sum of spectrums', fontsize=14)
    else:
        fig.suptitle(expt.nxs+': Spectrum number %g/%g'%(n,(len(expt.spectrums)-1)), fontsize=14)


    if expt.is_show_zooms:
        # Plot each peak
        colors = iter(['#006BA4', '#FF800E', '#ABABAB', '#595959', '#C85200', 'b', '#A2C8EC', '#FFBC79']*200)
        linestyles = iter(['-.', '-', ':']*400)

        count = 0
        for group in groups:
            for peak in group.peaks:
                if count%2==0: fig = plt.figure(figsize=(14,4.7))
                plt.subplot(1, 2, count%2+1)

                position = peak.position
                position_init = peak.position_init

                ind_min = np.argmin(np.abs(np.array(eVs)-0.9*position))
                ind_max = np.argmin(np.abs(np.array(eVs)-1.1*position))

                spectrum_zoom = spectrum[ind_min:ind_max]
                eVs_zoom = eVs[ind_min:ind_max]

                if dparams_list != None:
                    spectrum_fit_zoom = spectrum_fit[ind_min:ind_max]
                    intensity_rel = peak.intensity_rel
                    area = group.area_list[n]

                    if peak.is_fitpos:
                        title0 = 'position(init) = %g eV, position(fit)=%g eV'%(position_init,position)
                    else:
                        title0 = 'position = %g eV'%(position)

                    title = group.elem_name + ' ' + peak.name + '\n' \
                            +'group area = %g, relative int = %g'%(area,intensity_rel) + '\n'\
                            + title0
                else:
                    title = group.elem_name + ' ' + peak.name +'\n'+'position = %g eV'%(position)

                plt.gca().set_title(title)

                plt.plot(eVs_zoom, spectrum_zoom, 'k.')
                if dparams_list != None: plt.plot(eVs_zoom, spectrum_fit_zoom, 'r-', linewidth = 2)
                plt.xlabel('E (eV)')

                # Plot each line in the zoom
                for group_tmp in groups:
                    for peak_tmp in group_tmp.peaks:
                        position_tmp = peak_tmp.position
                        if (eVs[ind_min]<position_tmp and eVs[ind_max]>position_tmp):
                            if (group_tmp.name==group.name and peak_tmp.name == peak.name):
                                color = 'k'
                                linestyle = '--'
                            else:
                                color = next(colors)
                                linestyle = next(linestyles)

                            plt.axvline(x = position_tmp , label = group_tmp.elem_name+' '+peak_tmp.name,
                                        linestyle = linestyle, color = color)
                plt.legend()

                if  count%2==1: plt.show()
                count+=1

        # If there was an odd number of plots, add a blank figure
        if count%2==1:
            plt.subplot(122).axis('off')
        
    plt.show()
 

def Plot_fit_results(expt, spectrum_index=None, dparams_list=None, is_save=False):
    '''
    Plot all the params in dparams_list, as a function of the spectrum.
    
    Parameters
    ----------
    expt : object
        object from the class Experiment
    spectrum_index : int, optional
        index of the spectrum to be displayed
    dparams_list : array_like, optional
        list of parameters to fit    
    is_save : boolean, optional
         save each plot in a png if True
    '''

    groups = expt.groups
    spectrums = expt.spectrums

    scans = np.arange(np.shape(spectrums)[0])


    # Plot areas & save plots
    is_title = True
    elem_already_plotted = []
    for group in groups:

        if group.elem_name not in elem_already_plotted:

            elem_already_plotted = np.append(elem_already_plotted, group.elem_name)

            fig, ax = plt.subplots(figsize=(15,4))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))


            # To group elem on the same plot
            for group_tmp in groups:
                if group_tmp.elem_name == group.elem_name:
                    # Full list_lines_str is a problem when there are too many lines
                    if len(group_tmp.peaks)>1:
                        list_lines_str = '['+group_tmp.peaks[0].name[:1]+']'
                    else:
                        list_lines_str = '['+' '.join([p.name for p in group_tmp.peaks])+']'
                    plt.plot(scans, group_tmp.area_list, '.-', label = 'Area %s %s'%(group_tmp.elem_name,list_lines_str))

            plt.legend(bbox_to_anchor=(0,1.02,1,0.2),loc = 'lower left',ncol = 5)
            plt.tight_layout()

            if is_save: plt.savefig(expt.working_dir+expt.id+'/area_'+group.elem_name+'.png')
            if spectrum_index!=None: plt.axvline(x = spectrum_index, linestyle = '--', color = 'black')
            if is_title:
                ax.set_title('AREAS\n\n')
                ax.title.set_fontsize(18)
                ax.title.set_fontweight('bold')
                is_title = False

            #ax.set_ylim(bottom=ax.get_ylim()[0]*0.7, top=ax.get_ylim()[1]*1.3)
            plt.show()

    # Plot positions & save plots
    is_title = True
    for group in groups:
        for peak in group.peaks:
            if peak.is_fitpos:
                fig, ax = plt.subplots(figsize=(15,4))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
                plt.plot(scans, peak.position_list, 'b.-', label = 'Position %s '%(group.elem_name+'.'+peak.name))
                plt.legend()
                if is_save: plt.savefig(expt.working_dir+expt.id+'/position_'+group.elem_name+'_'+peak.name+'.png')
                if is_title:
                    ax.set_title('POSITIONS\n\n')
                    ax.title.set_fontsize(18)
                    ax.title.set_fontweight('bold')
                    is_title = False
                if spectrum_index!=None: plt.axvline(x = spectrum_index, linestyle = '--', color = 'black')
                plt.show()

    # Plot other params & save plots
    # Plot only the params which were fitted
    is_title = True
    for name in dparams_list:
        if name[:-5] in expt.list_isfit:
            fig, ax = plt.subplots(figsize=(15,4))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            plt.plot(scans, dparams_list[name], 'k.-', label = name[:-5])
            plt.legend()
            if is_save: plt.savefig(expt.working_dir+expt.id+'/'+str(name[:-5])+'.png')
            if is_title:
                ax.set_title('OTHER PARAMETERS\n\n')
                ax.title.set_fontsize(18)
                ax.title.set_fontweight('bold')
                is_title = False
            if spectrum_index!=None: plt.axvline(x = spectrum_index, linestyle = '--', color = 'black')
            plt.show()
            

def Choose_spectrum_to_plot(expt):
    '''
    Select a spectrum to plot with its fit.
    
    Parameters
    ----------
    expt : object
        object from the class Experiment
    '''
    
    def on_button_add_clicked(b):
        """Add the plot to the report."""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        expt.is_show_peaks = w_is_show_peaks.value
        expt.is_show_zooms = w_is_show_zooms.value
        expt.is_show_subfunctions = w_is_show_subfunctions.value

        code = 'FE.Treatment.Load_results(expt, spectrum_index='+str(w_index.value)+')'
        Utils.Create_cell(code=code, position='below', celltype='code', is_print=True, is_execute=True)

    def on_button_display_clicked(b):
        """Display the selected plot to the report"""

        # Clear the plots and reput the boxes
        clear_output(wait=True)
        Choose(expt)

        expt.is_show_peaks = w_is_show_peaks.value
        expt.is_show_zooms = w_is_show_zooms.value
        expt.is_show_subfunctions = w_is_show_subfunctions.value

        display(widgets.HBox([w_index, w_is_show_peaks, w_is_show_zooms, w_is_show_subfunctions, button_display]))
        display(button_add)

        # Plot the spectrum and fit
        Load_results(expt, w_index.value)


    w_index = widgets.IntText(description="Spectrum:",
                              style=style,
                              layout=widgets.Layout(width='200px'))

    w_is_show_peaks = widgets.Checkbox(
                              description='Show peaks?',
                              value=expt.is_show_peaks,
                              style=style,
                              layout=widgets.Layout(width='120px'))
    
    w_is_show_zooms = widgets.Checkbox(
                              value=expt.is_show_zooms,
                              layout=widgets.Layout(width='120px'),
                              style=style,
                              description='Show zooms?')       
        
    w_is_show_subfunctions = widgets.Checkbox(
                          description='Show sub-functions?',
                          value=expt.is_show_subfunctions,
                          style=style,
                          layout=widgets.Layout(width='170px'))

    button_display = widgets.Button(description="Preview the selected plot",layout=widgets.Layout(width='300px'))
    button_display.on_click(on_button_display_clicked)

    display(widgets.HBox([w_index, w_is_show_peaks, w_is_show_zooms, w_is_show_subfunctions, button_display]))

    button_add = widgets.Button(description="Add the selected plot",layout=widgets.Layout(width='300px'))
    button_add.on_click(on_button_add_clicked)
    display(button_add)

    
def Load_results(expt, spectrum_index=0):
    '''
    Load and plot the results of a previous fit.
    Redo the fit with all the results from FitResults.csv
    
    Parameters
    ----------
    expt : object
        object from the class Experiment
    spectrum_index : int, optional
        index of the spectrum to be displayed
    '''
    
    groups = expt.groups

    dparams_list = {'gain_list', 'eV0_list', 'sl_list', 'ct_list',
                    'sfa0_list', 'tfb0_list', 'twc0_list',
                    'noise_list', 'fano_list', 'epsilon_list',
                    'fG_list', 'fA_list', 'fB_list', 'gammaA_list', 'gammaB_list'
                    }

    # Init all the lists
    dparams_list = dict.fromkeys(dparams_list, np.array([]))

    for group in groups:
        group.area_list = np.array([])
        for peak in group.peaks:
            peak.intensity_rel_list = np.array([])
            peak.position_list = np.array([])


    with open(expt.working_dir+expt.id+'/FitResults.csv', "r") as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            for group in groups:
                group.area_list = np.append(group.area_list, np.float(row['#'+group.name+'.area'].replace(',','.')))

                for peak in group.peaks:
                    peak.position_list = np.append(peak.position_list,
                                       np.float(row['#'+group.elem_name+'_'+peak.name+'.position'].replace(',','.')))

            for name in dparams_list:
                    dparams_list[name] = np.append(dparams_list[name], np.float(row['#'+name[:-5]].replace(',','.')))


    print("Fit results for %s"%expt.nxs)
    print("Spectrum interval = [%g,%g]"%(expt.first_spectrum,expt.last_spectrum))
    print("Channel interval = [%g,%g]"%(expt.first_channel,expt.last_channel))
    tmp = np.array([0,1,2,3,4])
    print("List of chosen elements: ", ["Element %g"%g for g in tmp[expt.fluospectrums_chosen]])
    print("List of fitted parameters: "+str(expt.list_isfit))
    print("")
    print("Initial fit parameters:")
    print("beam energy = %g"%expt.beam_energy)
    print("gain = %g"%expt.gain +"; eV0 = %g"%expt.eV0)
    print("epsilon = %g"%expt.epsilon+"; fano = %g"%expt.fano+
          "; noise = %g"%expt.noise)
    print("sl = %g"%expt.sl+"; ct = %g"%expt.ct)
    print("sfa0 = %g"%expt.sfa0+"; tfb0 = %g"%expt.tfb0+"; twc0 = %g"%expt.twc0)
    print("fG = %g"%expt.fG+"; fA = %g"%expt.fA+"; fB = %g"%expt.fB+"; gammaA = %g"%expt.gammaA+"; gammaB = %g"%expt.gammaB)
    print("")


    # To generate pdf plots for the PDF rendering
    set_matplotlib_formats('png', 'pdf')

    Plot_spectrum(expt, spectrum_index, dparams_list)
    Plot_fit_results(expt, spectrum_index, dparams_list, is_save=False)

    # Restore it to png only to avoid large file size
    set_matplotlib_formats('png')    