import requests
import json
import os

def get_solar_power():
    # Create the directory if it doesn't exist
    os.makedirs('solar_power_json', exist_ok=True)

    url = "https://re.jrc.ec.europa.eu/api/v5_3/PVcalc"

    # lat = 28.6139   # Delhi latitude
    # lon = 77.2090   # Delhi longitude

    lat = 23.7957   # Dhanbad latitude
    lon = 86.4304   # Dhanbad longitude

    params = {
        'lat': lat,                  
        'lon': lon,                 
        'peakpower': 1,              # kW
        'loss': 14,                  # %
        'angle': 28,                 # degrees tilt
        'fixed': 1,
        'aspect': 0,                 # 0 = South facing
        'pvtechchoice': 'crystSi',   # Panel type
        'mountingplace': 'building',
        'outputformat': 'json',
        

    }

    response = requests.get(url, params=params)
    data = response.json()

    with open('solar_power_json/solar_data.json', 'a') as f:
        json.dump(data, f, indent=4)

    yearly_power = data['outputs']['totals']['fixed']['E_y']


    # Example: print yearly energy
    # print("Yearly Energy Production (kWh):", yearly_power )

    return yearly_power




    '''
    The API will return a JSON containing:

    E_d → daily average production (kWh/day)

    E_m → monthly totals (list of 12 months)

    E_y → yearly total (kWh/year)

    Radiation values: H(i_opt), etc.
    '''



    '''
    If you want hourly production, you need to use the seriescalc endpoint instead:

    bash
    Copy
    Edit
    https://re.jrc.ec.europa.eu/api/v5_2/seriescalc
    '''
