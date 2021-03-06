from . import Utils
from lib.extraction.common import PyNexus as PN

import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import time
import subprocess
import math

import shutil

"""Frontend library for all the widgets concerning the actions in the notebook."""

    
def Check_and_init(expt):
    '''
    Check if files and folders exist, then create the first cell.

    Parameters
    ----------
    expt : object
        object from the class Experiment
    '''

    print("Results will be saved in the folder:")
    if not os.path.exists(expt.working_dir):
        print(PN._RED+"Careful, the following folder does not exist and should be created:"+PN._RESET)
    print(expt.working_dir)
    print("")
        
    print("The original nexus files should be in the folder:")    
    if not os.path.exists(expt.recording_dir):
        print(PN._RED+"Careful, the following folder does not exist:"+PN._RESET)
    print(expt.recording_dir)
    print("")
    
    if not os.path.exists(expt.notebook_name):
        print(PN._RED+"Careful, assign the correct notebook name to expt.notebook_name."+PN._RESET)
        print("")

    file = 'parameters/peaks/Default_Parameters_peaks.csv'
    if not os.path.exists(file):
        print(PN._RED+"The following file does not exist:"+PN._RESET)
        print(file)
        print("This file contains the peaks to be displayed by default.")
        print("")
     
    file = 'parameters/extraction/Default_Parameters_extraction.csv'
    if not os.path.exists(file):
        print(PN._RED+"The following file does not exist:"+PN._RESET)
        print(file)
        print("This file contains the parameters for extraction by default.")
        print("")

    file = 'parameters/fit/Default_Parameters_fit.csv'
    if not os.path.exists(file):
        print(PN._RED+"The following file does not exist:"+PN._RESET)
        print(file)
        print("This file contains the parameters for extraction by default.")
        print("")        
        
    Utils.Create_cell(code='FE.Action.Choose(expt)', position ='at_bottom', celltype='code', is_print=False)

def Choose(expt):
    '''
    Choose the next action to do.

    Parameters
    ----------
    expt : object
        object from the class Experiment
    '''

    # Define the list of nxs files in the recording directory
    expt.list_nxs_files = [file for file in sorted(os.listdir(expt.recording_dir)) if 'nxs' in file][::-1]
    if expt.list_nxs_files == []:
        print(PN._RED+'There is no nexus file in the recording folder.'+PN._RESET)
        print(PN._RED+'Recording folder: %s'%expt.recording_dir+PN._RESET)
        expt.list_nxs_files = ['SIRIUS_NoFileFound_00_00_00.nxs']
        
    def selection_scan(nxs_file):
        """
        Called by the widget to select the scan to be treated.
        """

        # Generate several identifiers for the scan
        expt.nxs = nxs_file
        Define_scan_identifiers(expt)
        
        # Create a folder for saving params and results, if it does not already exist.
        if not os.path.exists(expt.working_dir+expt.id):
            os.mkdir(expt.working_dir+expt.id)

        # Check if the csv files for parameters already exists, if not copy the default ones
        if not os.path.isfile(expt.working_dir+expt.id+'/Parameters_extraction.csv'):
            shutil.copy('parameters/extraction/Default_Parameters_extraction.csv',
                        expt.working_dir+expt.id+'/Parameters_extraction.csv')

        if not os.path.isfile(expt.working_dir+expt.id+'/Parameters_fit.csv'):
            shutil.copy('parameters/fit/Default_Parameters_fit.csv',
                        expt.working_dir+expt.id+'/Parameters_fit.csv')            
            
    def on_button_treat_clicked(b):
        """
        Generate and execute cells corresponding to the chosen scan.
        """
        #clear_output(wait=False)
        
        selection_scan(w_print_scan.value)
        
        Utils.Create_cell(code='FE.Treatment.Choose(expt)',
                    position ='below', celltype='code', is_print=False)

    
        Utils.Create_cell(code='### '+expt.id, position ='below', celltype='markdown', is_print=True)
        
        Utils.Delete_current_cell()
        
    def on_button_refresh_clicked(b):
        """
        Re-execute the cell to update it.
        """
        
        Utils.Refresh_current_cell()
     
        
    def on_button_export_clicked(b):
        """
        Export the notebook to PDF.
        """
        
        print('Export in progress...')
        
        export_done = Export_nb_to_pdf(expt.notebook_name)
        
        if export_done:
            print('Notebook exported to %s.pdf'%expt.notebook_name.split('.')[0])
        else:
            print("There was something wrong with the export to pdf.")
            print("Did you rename the Notebook? If yes:")
            print("1) Change the value of expt.notebook_name in the first cell (top of the Notebook).")
            print("2) Re-execute the first cell.")
            print("3) Try to export the pdf again in the last cell (bottom of the Notebook).")

                
    def on_button_markdown_clicked(b):
        """
        Insert a markdown cell below the current cell.
        """ 
        
        Utils.Delete_current_cell()
        
        Utils.Create_cell(code='', position ='below', celltype='markdown', is_print=True, is_execute=False)
    
        Utils.Create_cell(code='FE.Action.Choose(expt)', position ='at_bottom', celltype='code', is_print=False)
        
    
    # Display the widgets
   
    # Click to treat a single scan
    button_treat = widgets.Button(description="Treat scan")
    button_treat.on_click(on_button_treat_clicked)
    
    # Click to refresh the list of files
    button_refresh = widgets.Button(description="Refresh")
    button_refresh.on_click(on_button_refresh_clicked)
    
    # Click to export to pdf
    button_export = widgets.Button(description="Export to PDF")
    button_export.on_click(on_button_export_clicked)
    
    # Click to insert a markdown cell
    button_markdown = widgets.Button(description="Insert comment")
    button_markdown.on_click(on_button_markdown_clicked)

    # Widget for selection of scan
    w_print_scan = widgets.Dropdown(
                    options=expt.list_nxs_files,
                    layout=widgets.Layout(width='400px'),
                    style={'description_width': 'initial'})
                    
    buttons0 = widgets.HBox([w_print_scan, button_treat])
    display(buttons0)
                    
    buttons1 = widgets.HBox([button_refresh, button_export, button_markdown])
    display(buttons1)
    

def Export_nb_to_pdf(nb_name):
    '''
    Export the notebook to pdf using a command line through the OS.

    Parameters
    ----------
    nb_name : str
        full name of the notebook. Ex: 'JupyLabBook.ipynb'

    Returns
    -------
    bool
        export_done, True if the export suceeded without error/warning
    '''
    
    # Save the current state of the notebook (including the widgets)
    Utils.Save_nb()
    
    t0 = time.time()
    rc = 1
    while rc>0:
        if (time.time()-t0) > 100:
            # Timeout before PDF export is considered as failed
            export_done = False
            break
        else:
            time.sleep(3)
            command = 'jupyter nbconvert '
            command+= nb_name
            command+= ' --to pdf '
            command+= ' --TagRemovePreprocessor.remove_cell_tags=\"[\'notPrint\']\" ' # Remove the widgets from the PDF
            command+= ' --no-input ' # Remove the code cells
            #command+= '--template latex_template.tplx' # Custom template (not working with nbconvert >= 6.0)
            rc = subprocess.call(command,shell=True)
            if rc==0: export_done = True
                
    return export_done




def Define_scan_identifiers(expt):
    '''
    Create a series of identifiers for the current scan.

    Parameters
    ----------        
    expt : object
        object from the class Experiment
    '''

    # For example:
    # expt.nxs = 'SIRIUS_2017_12_11_08042.nxs'
    # expt.path = '/Users/arnaudhemmerle/recording/SIRIUS_2017_12_11_08042.nxs'
    # expt.id = 'SIRIUS_2017_12_11_08042'
    # expt.number = 8042
    
    expt.path = expt.recording_dir+expt.nxs
    expt.id = expt.nxs[:-4]
    split_name = expt.nxs.split('.')[0].split('_')
    expt.number = int(expt.nxs.split('.')[0].split('_')[-1])
    
    # Sets the tokens
    expt.is_extract_done = False
    expt.is_set_peaks_done = False
    expt.is_fit_done = False
    



    