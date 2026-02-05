import baybe
import pandas as pd
from pathlib import Path
import os
import numpy as np

def prepare_campaign_df(
        campaign,
        dep_rate_calibration: float,
        Ipk_list: list[float] = None,
    ) -> pd.DataFrame:
    """
    Prepare a DataFrame from a BayBe Campaign object.

    Args:
        campaign: BayBe Campaign object.
        dep_rate_calibration: Calibration factor for deposition rate.
        Ipk_list: Optional list of Ipk values to add to the DataFrame.

    Returns:
        pd.DataFrame: Prepared DataFrame with computed columns and adjusted parameters.
    """
    df_campaign = campaign.measurements.copy()  # store campaign measurements into df

    # --- Add or compute columns only if they don't exist ---
    if 'Ipk (A)' not in df_campaign.columns and Ipk_list is not None:
        df_campaign['Ipk (A)'] = Ipk_list

    # Compute duty cycle only if the needed fields exist and it’s missing
    if 'Duty Cycle (ratio)' not in df_campaign.columns:
        if {'PRR (Hz)', 'PW (us)'}.issubset(df_campaign.columns):
            df_campaign['Duty Cycle (ratio)'] = df_campaign['PRR (Hz)'] * (df_campaign['PW (us)'] * 1e-6)

    # Compute PRR only if it’s missing but duty cycle exists
    if 'PRR (Hz)' not in df_campaign.columns:
        if {'Duty Cycle (ratio)', 'PW (us)'}.issubset(df_campaign.columns):
            df_campaign['PRR (Hz)'] = df_campaign['Duty Cycle (ratio)'] / (df_campaign['PW (us)'] * 1e-6)

    # --- Build parameter list dynamically ---
    params = [p.name for p in campaign.parameters]

    # Add only if not already included
    for add_param in ['Ipk (A)', 'PRR (Hz)', 'Voltage (V)', 'duty cycle']:
        if add_param in df_campaign.columns and add_param not in params:
            params.append(add_param)

    # Remove only if it exists
    for remove_param in ['PRR (Hz)', 'Duty Cycle (ratio)', 'pos. Delay (us)', 
                        'pos. PW (us)', 'pos. Setpoint (V)']:
        if remove_param in params:
            params.remove(remove_param)

    #Calibrate the deposition-rate, dep-rate calibration imported from dataset folder
    df_campaign['y1'] = df_campaign['y1']*1000*dep_rate_calibration

    return df_campaign

def get_ipk(
        path: Path,
        corr_time: float,
        campaign: baybe.Campaign,
    ) -> list[float]:
    """
    Extract Ipk values from the oscilloscope log file.

    Args:
        path: Path to the directory containing oscilloscope log files.
        corr_time: Time correction factor.
        campaign: BayBe Campaign object.

    Returns:
        List of Ipk values.
    """
    Ipk_list = []
    id = 0
    for file in os.listdir(path):
        #import relevant data from Oscci LogFiles
        df = pd.read_json(path / file)

        #pull df from campaign for PW determination
        df_campaign = campaign.measurements

        #Create adjusted time
        df['Time'] = np.linspace(0,int(len(df)*corr_time),len(df)) #modify x for the range specified in sofware and the #of sampled points (almost always 10,000)

        #Define Triggers
        trigger = 900*corr_time #est. where trigger is
        trigger_exclude = 250*corr_time #est. of # of samples to exclude post-trigger from Ipk determination 
        cutoff_trigger = trigger + trigger_exclude
        cutoff_pulse = cutoff_trigger + df_campaign['PW (us)'][id] #add pulse-width to determination zone, avoids subseuqnet oscillations getting logged as Ipk

        #use defined triggers for Ipk determination
        df_mask = df.loc[((df['Time']> cutoff_trigger) & (df['Time'] < cutoff_pulse))] #finds peak-current within shaded green area, currently based on sample count, should be adjusted to pulse-width in the future
        
        Ipk_list.append(np.max(df_mask[2])/(45.58)) #units [A/cm2] - peak current density correction for 3" target

    return Ipk_list