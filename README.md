# Big G Express: Predicting Derates

In this project, you will be working with fault code data and vehicle onboard diagnostic data to try and predict an upcoming full derate. These are indicated by an SPN 5246.

You have been provided with a two files containing the data you will use to make these predictions (J1939Faults.csv and VehicleDiagnosticOnboardData.csv) as well as two files describing some of the contents (DataInfo.docx and Service Fault Codes_1_0_0_167.xlsx)

Note that in its raw form the data does not have "labels", so you must define what labels you are going to use and create those labels in your dataset. Also, you will likely need to perform some significant feature engineering in order to build an accurate predictor.

Additional cleaning tasks:

* Remove faults occurring in the vicinity of the service locations at (36.0666667, -86.4347222), (35.5883333, -86.4438888), and (36.1950, -83.174722)
  * When being worked on, diagnostics can throw out several codes at the service location.  Filter within .25 to 1 mile to exclude this scenario.
* Remove faults where the EquipmentID has more than 5 characters.

Goal

- Focus on full derates.  When one happens, it can cost the company upwards of $4k to get towed.
- What can happen beforehand to keep the truck from getting towed.
- Look for SPN codes 5246 (BAD) and 1569 (Not as bad, but bad.)

  - A 75% derate (SPN = 1569, FMI = 31) Reduces engine torque by 25%.
  - A idle level derate (SPN = 5246) will require a tow.

  ## NOTES
- Time Windows to look at:

  - Rohit says Play around with time windows under 1 week.  He also says that it would be a good ida to use "rolling windows".  Rolling windows are your friend.
    - A rolling window, also known as a moving window, is a technique used in data analysis and signal processing to analyze data over a fixed window size that "rolls" or "moves" through the data over time. It's essentially a way to apply a function or computation to a sliding window of data points in a time series or other ordered data.
    - Pandas provides a `rolling()` function that can be used to apply a rolling window calculation to a DataFrame or Series object. For example, you can use `df.rolling(window=10).mean()` to calculate a rolling average over a window size of 10 for a DataFrame `df`.
  - Think carefully about how you will partition your train, validation, and test datasets. Since you’re dealing with time series data, you don’t want to leak information about the future during model training
- faults_diagnostics = gpd.GeoDataFrame(faults_diagnostics, geometry = gpd.points_from_xy(faults_diagnostics.Longitude, faults_diagnostics.Latitude))
- faults_diagnostics = faults_diagnostics[~(faults_diagnostics.distance(Point(-86.4347222, 36.0666667)) < 0.01)]
- SPN (suspect paremeter number) - This tells you where the problem is happening
- FMI - (Failure Mode Indicator) tells you what the problem is

| SPN Code | Description                                                                                                          |
| -------- | -------------------------------------------------------------------------------------------------------------------- |
| 31       | Exhaust Gas Recirculation (EGR) Valve Position                                                                       |
| 84       | Engine Torque                                                                                                        |
| 91       | Accelerator Pedal Position                                                                                           |
| 97       | Water in Fuel Indicator                                                                                              |
| 108      | Engine Coolant Temperature                                                                                           |
| 120      | Oil Pressure                                                                                                         |
| 123      | Boost Pressure                                                                                                       |
| 305      | Exhaust Gas Temperature 1                                                                                            |
| 356      | Aftertreatment Diesel Particulate Filter Differential Pressure                                                       |
| 411      | Engine Exhaust Gas Recirculation (EGR) Mass Flow Rate                                                                |
| 436      | Engine Oil Pressure Low                                                                                              |
| 524      | Aftertreatment 1 Diesel Exhaust Fluid Dosing Unit Temperature                                                        |
| 574      | Engine Exhaust Gas Recirculation (EGR) Temperature                                                                   |
| 636      | Engine Turbocharger 1 Compressor Intake Pressure                                                                     |
| 651      | Engine Turbocharger 1 Speed                                                                                          |
| 659      | Engine Fuel Injection Timing                                                                                         |
| 677      | Diesel Exhaust Fluid (DEF) Tank Level                                                                                |
| 678      | Diesel Exhaust Fluid (DEF) Pressure                                                                                  |
| 1001     | Engine Air Inlet Pressure                                                                                            |
| 102      | Intake Manifold Pressure                                                                                             |
| 1080     | Engine Fuel Flow Rate                                                                                                |
| 171      | Exhaust Back Pressure                                                                                                |
| 265      | Fuel Rail Pressure                                                                                                   |
| 512      | Aftertreatment 1 Exhaust Gas Temperature Sensor 3                                                                    |
| 513      | Aftertreatment 1 Diesel Oxidation Catalyst System Efficiency                                                         |
| 515      | Aftertreatment 1 Diesel Particulate Filter System Efficiency                                                         |
| 524      | Aftertreatment 1 Diesel Exhaust Fluid Dosing Unit Temperature                                                        |
| 531      | Aftertreatment 1 Diesel Particulate Filter Intake Gas Temperature                                                    |
| 583      | Aftertreatment 1 Particulate Matter Trap Differential Pressure                                                       |
| 596      | Aftertreatment 1 Diesel Particulate Filter Regeneration Request                                                      |
| 597      | Aftertreatment 1 Diesel Particulate Filter Regeneration Inhibit Switch                                               |
| 599      | Aftertreatment 1 Diesel Particulate Filter System Active Regeneration Inhibit Reason                                 |
| 601      | Aftertreatment 1 Diesel Particulate Filter System Active Regeneration Request                                        |
| 602      | Aftertreatment 1 Diesel Particulate Filter System Active Regeneration Status                                         |
| 611      | Aftertreatment 1 Diesel Exhaust Fluid (DEF) Quantity                                                                 |
| 612      | Aftertreatment 1 Diesel Exhaust Fluid (DEF) Air Pressure                                                             |
| 698      | Aftertreatment 1 Diesel Exhaust Fluid System Fault                                                                   |
| 799      | Aftertreatment 1 Diesel Particulate Filter Differential Pressure Sensor, Signal Not Plausible                        |
| 1053     | Diesel Particulate Filter (DPF) Pressure Sensor Circuit - Voltage Above Normal or Shorted to High Source             |
| 1054     | Diesel Particulate Filter (DPF) Pressure Sensor Circuit - Voltage Below Normal or Shorted to Low Source              |
| 1055     | Diesel Particulate Filter (DPF) Pressure - Data Valid But Above Normal Operational Range - Moderately Severe Level   |
| 1056     | Diesel Particulate Filter (DPF) Pressure - Data Valid But Above Normal Operational Range - Most Severe Level         |
| 1057     | Diesel Particulate Filter (DPF) Pressure - Data Valid But Below Normal Operational Range - Moderately Severe Level   |
| 1058     | Diesel Particulate Filter (DPF) Pressure - Data Valid But Below Normal Operational Range - Most Severe Level         |
| 1061     | Aftertreatment Diesel Particulate Filter Differential Pressure - Data Valid But Below Normal Operating Range - Moder |
