import numpy as np
import pandas as pd 

cfs_to_afd = 2.29568411*10**-5 * 86400
afd_to_cfs = 1 / cfs_to_afd

def water_day(d, is_leap_year):
    # Convert the date to day of the year
    day_of_year = d.timetuple().tm_yday
    
    # For leap years, adjust the day_of_year for dates after Feb 28
    if is_leap_year and day_of_year > 59:
        day_of_year -= 1  # Correcting the logic by subtracting 1 instead of adding
    
    # Calculate water day
    if day_of_year >= 274:
        # Dates on or after October 1
        dowy = day_of_year - 274
    else:
        # Dates before October 1
        dowy = day_of_year + 91  # Adjusting to ensure correct offset
    
    return dowy

def split_return(x,match):
    spl_tex = x.split('.')
    out = False
    if spl_tex[0] == match:
        out = True
    return out




