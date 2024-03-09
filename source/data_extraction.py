import requests
import pandas as pd

def get_EIA_natgas_data(fname, token = None, api_call = None):

    if not token:
        try:
            df = pd.read_csv(fname)
            return df
        except:
            raise ValueError('Expected non-None value(s) for token, api_call, and params!')
    
    else:
        if not api_call:
            api_call =\
        "https://api.eia.gov/v2/natural-gas/stor/wkly/data/?frequency=weekly&data[0]=value&facets[series][]=NW2_EPG0_SWO_R48_BCF&sort[0][column]=period&sort[0][direction]=asc&api_key={}&out=json"

        res = requests.get(
            api_call.format(token)
        )
        df = pd.DataFrame(res.json()['response']['data'])

        df.to_csv(fname, index = False)

        return df