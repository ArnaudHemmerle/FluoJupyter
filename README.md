# FluoJupyter

FluoJupyter is a Jupyter Notebook to analyze X-Ray Fluorescence (XRF) experiments on the beamline SIRIUS at the synchrotron SOLEIL.  
The notebook should be first set up by an Expert following the instructions in the "Expert" section. User can then follow the guidelines in the "User" section to start using the notebook.

## User

### Getting Started
1. Once the notebook is open, run the first cell with the parameters starting with:
```
# Run this cell
from FluoJupyter_functions import *
print_version()

########### EXPERT PARAMETERS ################
...
```
2. Run the next cell:
```
# Run this cell
generate_cells_on_click()
```
3. Click on the button 'Click me to create the next cells!'

### Enter the information on the experiment
1. Double-click on the text cells to enter your sample name, and add relevant description.

2. Enter the file name and click on the extraction button.

3. Change the values of 'first/last spectrum' and 'first/last channel' if needed. Re-extract the file.

### Choose the peaks
1. Run the two next cells:
```
# Run this cell
expmt.define_peaks()
```
```
# Run this cell
expmt.extract_elems()
w = widgets.interact(expmt.display_peaks,spectrum_index=widgets.IntText(value=0, step=1, description='Spectrum:'))
```

2. Modify the table to add/remove peaks and fit their position or not. You can also leave a peak in the list and do not include it in the analysis by writting no in the column '#Fit Peak?'. 

**Validate the sheet** by clicking outside it and **running the cell** expmt.extract_elems().

3. You can also directly edit the Excel file in the folder with you file name.

### Fit the spectrums
When you are done with the peak definition, run the cell:
```
# Run this cell
expmt.fit_spectrums()
```

You can follow the fit in real time. The results are continuously updated in the file FitResult.csv that you can open with you prefered software (Excel, Origin, Matlab ...).
When the fit is done, the fitted parameters will be displayed.

### Show the results on a specific spectrum
(Optional) If you want to check the results of the fit on a specific spectrum you can run the cell:
```
# Run this cell
w = widgets.interact(expmt.load_results,spectrum_index=widgets.IntText(value=0, step=1, description='Spectrum:'),is_save = widgets.Checkbox(value=False, description='Save Fit?'))
```
To save the fitting curve, check the box 'Save Fit?'.

### Continue with the next sample
To generate the cells for the next sample, click on the button at the bottom of the notebook.

## Expert

### Getting Started

Copy the files FluoJupyter.ipynb, FluoJupyter_functions.py, PyNexus.py and DefaultPeaks.csv on the local folder where the nxs files are. The nxs files should contain 'Fluo' in their name. Note that the notebook does not work with JupyterLab in its current version.

The aim of the Expert part is to determine the parameters in the first cell. It can be quite painful, but those parameters should not vary during an experiment. It is also possible to copy directly the parameters from a previous experiment (or from the examples provided here), and test if they are good for your current experimental setup.

### Conversion channels <-> eVs
First update the parameters 'gain' and 'eV0':
```
dparams_general['gain'] = 10.
dparams_general['eV0'] = 0.
```
used to convert the channels in eVs through the relation:
```
eVs = gain*channels + eV0
```

### Determine the list of peaks
1. Run the cell with the pameters, and the cells with the command:
```
generate_cells_on_click()
```
2. Click on the button.

3. Enter the required information in the widgets generated by the cell (or just try with the default values):  
```
expmt = Experiment(dparams_general, dparams_fit)
```
4. Extract about 10 spectrums from the file, and try to choose the nicest one (without any weird behavior). Choose a range of channels covering from the lowest peak you want to fit to the end of the elastic peak.


If you want to change the default values displayed in the box when you later reload the experiment, update the parameters:
```
dparams_general['ind_first_spectrum'] = 0
dparams_general['ind_last_spectrum'] = 1
dparams_general['ind_first_channel'] = 0
dparams_general['ind_last_channel'] = 2048
dparams_general['is_elements'] = [False, False, False, False, True]
```

**Warning:** Any time you update a value from the first cell (in dparams_general or dparams_fit), you need to re-execute the cell, and then to reload the experiment by executing the cell:
```
expmt = Experiment(dparams_general, dparams_fit)
```

5. Run the cell 
```
expmt.define_peaks()
```
If you do not have ipysheet installed, update directly the file Peaks.csv in the experiment folder.
List all the peaks you want to display. Keep the line name 'El' for the elastic peak and 'Co' for the Compton peak.
Validate the list by clicking outside the table and running the cell:
```
expmt.extract_elems()
```
**Warning:** For determining the fit parameters, you should fit the peak position of the most intense peaks.

### Determine the fit parameters
See inside FluoJupyter_functions.py to have an explanation on the peak fitting. Some parameters do not have to be fitted, and can be kept constant in our experimental configuration:
```
dparams_fit['fan'] = 0.12
dparams_fit['epsilon'] = 0.0036
dparams_fit['tfb1'] = 1e-10
dparams_fit['fA'] = 1.0e-10
dparams_fit['fB'] = 1.0e-10
dparams_fit['gammaA'] = 1.0e10
dparams_fit['gammaB'] = 1.0e10
```
Here we detail a procedure which seems to be robust for determining the other parameters. 
1. First start with these values :
```
dparams_fit['noise'] = 0.1
dparams_fit['sl'] = 0.
dparams_fit['ct'] = 0.
dparams_fit['tfb0'] = 0.1
dparams_fit['twc0'] = 1.
dparams_fit['twc1'] = 0.1
dparams_fit['fG'] = 1.5
dparams_fit['sfa0'] = 1e-10
dparams_fit['sfa1'] = 1e-5
```
2. Fit 'noise' and 'fG'
 - Update the parameter 'list_isfit':
```
dparams_general['list_isfit'] = ['sl','ct', 'noise', 'fG']
```
Reload the parameters and run all the cells up to (included):
```
expmt.fit_spectrums()
```
 - When the fit is done, get the average results for 'noise' and 'fG' using the command:
```
print(np.mean(expmt.dparams_list['noise_list']))
print(np.mean(expmt.dparams_list['fG_list']))
```
 - Update the parameters 'noise' and 'fG' in the first cell:
```
dparams_fit['noise'] = 0.11135974573713625
dparams_fit['fG'] = 1.4326013905627302
```
3. Repeat step 2 for 'tfb0' with: 
```
dparams_general['list_isfit'] = ['sl','ct','tfb0']
```
4. Repeat step 2 for 'twc0' with: 
```
dparams_general['list_isfit'] = ['sl','ct','twc0']
```
5. Repeat step 2 for 'twc1' with: 
```
dparams_general['list_isfit'] = ['sl','ct','twc1']
```
6. Repeat step 2 for 'sfa0' and 'sfa1' with: 
```
dparams_general['list_isfit'] = ['sl','ct','sfa0', 'sfa1']
```
7. Repeat step 2 for 'sl','ct', 'noise', 'fG' with: 
```
dparams_general['list_isfit'] = ['sl','ct', 'noise', 'fG']
```

### Finishing
It seems better to keep dparams_general['list_isfit'] = ['sl','ct', 'noise', 'fG'] for the User, especially if the intensity of the peaks varies significantly during an experiment.

Replace the file DefaultPeaks.csv by the file Peaks.csv. 

Delete the cells you have generated, and the file is ready for the User.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
