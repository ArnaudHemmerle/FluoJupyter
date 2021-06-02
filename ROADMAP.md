# Roadmap

Definition of what needs to be done for v1.0

## Software architecture
- Rewrite the architecture so that the scan does not have to be extracted every time a parameter is changed. For example, make a clear distinction between parameters for extraction and those for fitting. 
- Explore the idea of using panels instead of the current linear list of widgets. 
- Avoid the warning messages from the widgets when the notebook is restarted.
- Write a documentation with readthedocs for using the program, and for the functions. Include a tutorial with the effect of each fitting parameter on the peaks/spectrum.
- Think about having a Unit testing for controlling that each improvement does not include new bugs (hard to do with the GUI?)


## Visualization & data treatment
- Improve the display of plots when many peaks are involved. Do not display the weakest ones?
- Replace the parameter ```strength_min``` with a visual selection of which lines should be included in the model for each peak. 
- Allow the user to play with the initial fit parameters, with an imediate display of the curve (i.e. make a simulation with the current set of parameters).
- Allow the user to visualize the fit results as a function of the other parameters of the scan (in the spirit of 1Dplots in JupyLabBook)
- Rethink the way the results are saved. It should be easier for users to open them with other applications.
- Allow the user to easily import and visualize results from a previous fit, without doing the fit again. 
- Warn the users if previous results will be erased.
- Update the results in real time during the fit?


## Physics of the model
- How can we better model the escape peaks? Is there any litterature on the topic? Discussion with our detector group?
- Can we think about a *test sample* with known peaks that we could use to fix some of the parameters?
- Merge properly peaks from the model when they are close in energy.

