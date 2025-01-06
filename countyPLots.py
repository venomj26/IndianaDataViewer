#%%
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
# matplotlib inline
#%pip install seaborn
# import seaborn as sns
from shapely.geometry import Point, Polygon
import numpy as np
#%pip install googlemaps
import googlemaps
from datetime import datetime
plt.rcParams["figure.figsize"] = [8,6]
import pandas as pd
#%pip install  simplekml
import simplekml
import os
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from rasterio.crs import CRS
# %pip install rioxarray
# %pip install earthpy
import rioxarray as rxr
import earthpy as et
import rasterio
from rasterio.plot import show
import fiona
import glob
from shapely.geometry import box
# import seaborn as sns
# import seaborn as sns
import os
#from whitebox.whitebox_tools import WhiteboxTools
import rasterio
from rasterio.plot import show
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.colors as colors#
import requests
import urllib.parse
import re
#%%

# Step 1: Create a dictionary of state abbreviations and FIPS codes
state_abbr_map = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO',
    '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI',
    '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY',
    '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
    '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
    '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH',
    '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC', '46': 'SD',
    '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
    '54': 'WV', '55': 'WI', '56': 'WY', '72': 'PR', '78': 'VI'
}
# Step 1: Fetch the data from the website
url = 'https://www.nass.usda.gov/Data_and_Statistics/County_Data_Files/Frequently_Asked_Questions/county_list.txt'
response = requests.get(url)

# Step 2: Check if the request was successful
if response.status_code == 200:
    # Step 3: Parse the response content
    text_data = response.text

    # Step 4: Use regular expressions to capture relevant data
    # Adjusting regex to match the format in the response
    pattern = r"(\d{2})\s+(\d{2})\s+(\d{3})\s+([A-Za-z\s]+)\s+(\d)"
    matches = re.findall(pattern, text_data)

    if matches:
        # Step 5: Convert the matches to a DataFrame
        US_state_Codes = pd.DataFrame(matches, columns=['State', 'District', 'County', 'CountyName', 'Status'])

        # Step 6: Add a "state" column based on the District and County codes
        US_state_Codes['IsState'] = US_state_Codes.apply(lambda x: True if x['District'] == '00' and x['County'] == '000' else False, axis=1)
        US_state_Codes['StateName'] = US_state_Codes.apply(lambda x: x['CountyName'].strip() if x['IsState'] else None, axis=1)

        # Step 7: Forward-fill the StateName to the corresponding counties
        US_state_Codes['StateName'] = US_state_Codes['StateName'].ffill()

        # Step 8: Remove the rows that are states, leaving only county rows
        US_state_Codes = US_state_Codes[US_state_Codes['IsState'] == False].reset_index(drop=True)
        US_state_Codes['StateAbbr'] = US_state_Codes['State'].map(state_abbr_map)
        # Step 9: Display or save the resulting DataFrame
        #US_state_Codes.to_csv("county_data_with_states.csv", index=False)
        print("Data saved to county_data_with_states.csv")
    else:
        print("No matches found with the regex pattern.")
else:
    print("Failed to retrieve the data")


#%%
def sanitize_input(user_input):
    # Remove any single quotes and backslashes to prevent SQL injection
    return re.sub(r"[\'\"\\]", "", user_input)

def fetch_ssurgo_data_by_county(county_code_input):
    county_name = sanitize_input(county_code_input)
    url = 'https://sdmdataaccess.nrcs.usda.gov/tabular/post.rest'

    # SQL query without comments and in a single line
    #query = f"""SELECT musym, muname, mukey FROM legend l INNER JOIN mapunit mu ON l.lkey = mu.lkey WHERE areasymbol = 'IN079'"""
    #query = f"""SELECT compname,muareaacres FROM mupolygon mu LEFT OUTER JOIN component co ON mu.mukey = co.mukey WHERE areasymbol = 'IN079'"""
    query = f"""SELECT TOP 2  co.drainagecl, SUM(mu.muareaacres) AS total_muareaacres
                FROM mupolygon mu
                LEFT OUTER JOIN component co ON mu.mukey = co.mukey
                WHERE mu.areasymbol = '{county_code_input}'
                GROUP BY co.drainagecl
                ORDER BY total_muareaacres DESC"""
    
    # URL-encode the query
    encoded_query = urllib.parse.quote_plus(query)

    # Prepare the data as a URL-encoded string
    data = f"query={encoded_query}&format=JSON"

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = None
    try:
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print('An error occurred:', e)
        if response is not None:
            print('Server Response:', response.text)
        return None

# Fetch the SSURGO data for a specific county
county_code_input = "IN025"  # Replace with your desired county
ssurgo_data = fetch_ssurgo_data_by_county(county_code_input)
print(ssurgo_data)
# Process and display the data
if ssurgo_data:
    # Convert the 'Table' key data to a DataFrame
    df = pd.DataFrame(ssurgo_data['Table'], columns=['compname', 'areaacres'])
    print(df)
else:
    print('No SSURGO data found for the specified county.')

# %%
# Remove the \t characters from the CountyName column
US_state_Codes['CountyName'] = US_state_Codes['CountyName'].str.replace(r'\t+', '', regex=True).str.strip()

# Filter for Indiana (State FIPS Code: '18')
indiana_df = US_state_Codes[US_state_Codes['StateName'] == 'Indiana']

# Create a dictionary of county names and their corresponding county codes
indiana_county_dict = dict(zip(indiana_df['County'], indiana_df['CountyName']))
# Reverse dictionary to map county names to their codes
county_code_map = {v: f"IN{str(k).zfill(3)}" for k, v in indiana_county_dict.items()}

# %%
# Sanitize input to remove any unwanted characters
def sanitize_input(user_input):
    return re.sub(r"[\'\"\\]", "", user_input)

# Function to fetch data for a given county code
def fetch_ssurgo_data_by_county(areasymbol):
    url = 'https://sdmdataaccess.nrcs.usda.gov/tabular/post.rest'

    # SQL query to fetch compname and muareaacres from the mupolygon and component tables
    query = f"""SELECT TOP 2 co.drainagecl, SUM(mu.muareaacres) AS total_muareaacres
                FROM mupolygon mu
                LEFT OUTER JOIN component co ON mu.mukey = co.mukey
                WHERE mu.areasymbol = '{areasymbol}'
                GROUP BY co.drainagecl 
                ORDER BY total_muareaacres DESC"""

    # URL-encode the query
    encoded_query = urllib.parse.quote_plus(query)

    # Prepare the data for the POST request
    data = f"query={encoded_query}&format=JSON"

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = None
    try:
        # Send the POST request
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Return the JSON data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred for {areasymbol}: {e}")
        return None

# Initialize an empty DataFrame to store results
Aggregate_County = pd.DataFrame(columns=["county", "compname","county_code", "total_muareaacres"])

# Loop through each county in the dictionary
for key, county in indiana_county_dict.items():
    # Format the county code to be like "IN001", "IN002", etc.
    county_code = f"IN{str(key).zfill(3)}"
    #print("county_code: ",county_code)
    # Fetch data for the county
    ssurgo_data = fetch_ssurgo_data_by_county(county_code)
    #print("ssurgo_data: ",ssurgo_data)
    # Check if data was returned
    if ssurgo_data and 'Table' in ssurgo_data:
        for row in ssurgo_data['Table']:
            # Add each row of data to the DataFrame
            #print("row output: ",row)
            if row[0] is None or "None" in row[0]:
                #print("Nonetype encountered")
                continue
            else:
                Aggregate_County = pd.concat([Aggregate_County, pd.DataFrame([{"county": county,"county_code":county_code, "compname": row[0], "total_muareaacres": row[1]}])], ignore_index=True)
                #print("added")
                break #this statement is stopping the entry of multiple rows for each county.
print(Aggregate_County.head(2))
#Aggregate_County["county_code"] = Aggregate_County["county"].map(county_code_map)
# Save the DataFrame to a CSV file
#df.to_csv("indiana_soil_data.csv", index=False)
# Find the component with the maximum area for each county
#%%
max_area_df = Aggregate_County.loc[Aggregate_County['total_muareaacres'].idxmax()]
min_area_df = Aggregate_County.loc[Aggregate_County['total_muareaacres'].idxmin()]
print("Data fetches description max and min: ", max_area_df,min_area_df)
# %%


# def get_corn_yield(api_key, year):
#     url = 'https://quickstats.nass.usda.gov/api/api_GET/'
    
#     params = {
#         'key': api_key,
#         'commodity_desc': 'CORN',
#         'year': year,
#         'state_alpha': 'IN',  # Indiana
#         'sector_desc': 'CROPS',
#         'group_desc': 'FIELD CROPS',
#         'statisticcat_desc': 'YIELD',
#         'agg_level_desc': 'COUNTY',
#         'unit_desc': 'BU / ACRE',
#         'format': 'JSON'
#     }

#     response = requests.get(url, params=params)
#     data = response.json()

#     # Convert the data to a pandas DataFrame
#     df = pd.DataFrame(data['data'])
#     print(df.head(2))
#     df['county_name'] = df['county_name'].str.title()
#     # Map county names to county codes
#     df['county_code'] = df['county_name'].map(county_code_map)

#     # Select and return the required columns
#     return df[['county_name', 'county_code', 'Value']]

# # Example usage
# api_key = '80B6C46A-B786-31C0-8CE3-9BFB5ED50C32'
# year = 2020  # Specify the year
# corn_yield_data = get_corn_yield(api_key, year)
# #print(corn_yield_data)
# corn_yield_data['county_name'] = corn_yield_data['county_name'].str.lower()  # Convert to lowercase
# corn_yield_data['county_name'] = corn_yield_data['county_name'].str.replace(r'[^\w]', '', regex=True)

#%%
import requests
import pandas as pd

# API Key (replace with your actual key)
api_key = '80B6C46A-B786-31C0-8CE3-9BFB5ED50C32'

# Set the range of years you want to query
years = range(2000, 2023)  

# URL for the NASS Quick Stats API
api_url = 'https://quickstats.nass.usda.gov/api/api_GET/'

# Initialize an empty DataFrame to store the data
IN_cornYield = pd.DataFrame()

# Loop through the range of years and make the API request for each year
for year in years:
    params = {
        'key': api_key,
        'commodity_desc': 'CORN',
        'year': year,  # Query one year at a time
        'state_alpha': 'IN',  # Indiana
        'sector_desc': 'CROPS',
        'group_desc': 'FIELD CROPS',
        'statisticcat_desc': 'YIELD',
        'agg_level_desc': 'COUNTY',
        'unit_desc': 'BU / ACRE',
        'format': 'JSON'
    }

    # Make the API request for the current year
    response = requests.get(api_url, params=params)

    if response.status_code == 200:
        data = response.json()['data']  # Extract the 'data' field from the JSON response

        # Convert the data to a DataFrame and append it to the main DataFrame
        df_year = pd.DataFrame(data)
        IN_cornYield = pd.concat([IN_cornYield, df_year], ignore_index=True)

        print(f"Data for {year} appended. Total records: {len(IN_cornYield)}")

    else:
        print(f"Failed to retrieve data for {year}: {response.status_code}")

# Display the first few rows of the DataFrame
print(IN_cornYield.head())

IN_cornYield['county_name'] = IN_cornYield['county_name'].str.lower()  # Convert to lowercase
IN_cornYield['county_name'] = IN_cornYield['county_name'].str.replace(r'[^\w]', '', regex=True)


IN_cornYield_pivot = IN_cornYield.pivot_table(index=('county_name','county_code'), columns='year', values='Value', aggfunc='first').reset_index()



#%%
corn_yield_data=IN_cornYield_pivot[["county_name","county_code",2020]].copy()









# %%
fp=r"D:\OneDrive - purdue.edu\AutomatingAgFieldTrials\DOE_code\summer2021\DOE_ag\County_Boundaries_of_Indiana_2023\County_Boundaries_of_Indiana_2023.shp"
#fp=r"/Users/jhasneha/Documents/DOEjha/CodesResults/County_Boundaries_of_Indiana_2023"
indiana_counties_shape = gpd.read_file(fp)
indiana_counties_shape.plot()


# Apply the simplify function to reduce the number of points
# The 'tolerance' parameter controls the degree of simplification; a higher value removes more points.
indiana_counties_shape['geometry'] = indiana_counties_shape['geometry'].simplify(tolerance=0.01, preserve_topology=True)

indiana_counties_shape.plot()

# %%
indiana_counties_shape["county_fip"] = indiana_counties_shape["county_fip"].apply(
    lambda x: "IN" + str(x).split("18")[1].replace("\n     ", "") if "18" in str(x) else x
)
indiana_counties_shape["name"]=indiana_counties_shape["name"].apply(
    lambda x: x.replace("County", "") if "County" in str(x) else x
)
indiana_counties_shape['name'] = indiana_counties_shape['name'].str.lower()  # Convert to lowercase
indiana_counties_shape['name'] = indiana_counties_shape['name'].str.replace(r'[^\w]', '', regex=True)
# %%
indianaPlot=indiana_counties_shape[['county_fip',"name",'geometry']].copy()
# %%
Aggregate_County['county'] = Aggregate_County['county'].str.lower()  # Convert to lowercase
Aggregate_County['county'] = Aggregate_County['county'].str.replace(r'[^\w]', '', regex=True)

indianaPlot=indianaPlot.merge(Aggregate_County, left_on='name', right_on='county', how='left')
# %%
indianaPlot=indianaPlot.merge(corn_yield_data, left_on='county', right_on='county_name', how='left')

#%%
# Define the desired order of drainage classes from light to dark
drainage_order = ['ED', 'SED', 'WD', 'MWD', 'SPD', 'PD', 'VPD']

# Define the drainage class to abbreviation mapping
dcl_dict = {
    "Very poorly drained": "VPD",
    "Poorly drained": "PD",
    "Somewhat poorly drained": "SPD",
    "Moderately well drained": "MWD",
    "Well drained": "WD",
    "Somewhat excessively drained": "SED",
    "Excessively drained": "ED"
}
dcl_dict_rev = {v: k for k, v in dcl_dict.items()}

# Map the 'compname' to the drainage class abbreviations
indianaPlot["drainageclabbrvt"] = indianaPlot["compname"].map(dcl_dict)




# %%
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})


# Generate a color palette with exactly 7 colors (one for each drainage class) and assign them to the sorted classes
colors = sns.color_palette("copper", len(drainage_order))  # Generate the colors from light to dark
color_mapping = dict(zip(drainage_order, colors))  # Create a mapping of drainage class to color

# Now create a colormap using the sorted colors
color_list = [color_mapping[cls] for cls in drainage_order]
color_discrete = mcolors.ListedColormap(color_list)

# Ensure the values in 'drainageclabbrvt' are ordered to match the color order
indianaPlot['drainageclabbrvt'] = pd.Categorical(indianaPlot['drainageclabbrvt'], categories=drainage_order, ordered=True)

# Plot the data with the sorted colormap
myax = indianaPlot.plot(column='drainageclabbrvt', cmap=color_discrete, legend=True, legend_kwds={"title": "DCL",'loc': 'center left', 'bbox_to_anchor': (1, 0.5)})

# Overlay the Indiana county boundaries with transparency
indiana_counties_shape.plot(ax=myax, color="None", edgecolor="gray", alpha=0.4)
for legend_handle in myax.get_legend().legend_handles:
    legend_handle.set_markeredgewidth(0.5)
    legend_handle.set_markeredgecolor("black")
for spine in plt.gca().spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.xticks([])
plt.yticks([])
#plt.savefig(f"D:\OneDrive - purdue.edu\AutomatingAgFieldTrials\Fall2024\IndianaCountyPlots\indiana_soil_drainage_classes.png", bbox_inches='tight', dpi=600)
#plt.savefig(f"/Users/jhasneha/Documents/DOEjha/CodesResults/CountyIndiana/indiana_soil_drainage_classes.png", bbox_inches='tight', dpi=600)

plt.show()




# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
plt.style.use('dark_background')
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})

# # Get the unique categories from the 'compname' column
# unique_categories = indianaPlot['compname'].unique()
# colors = sns.color_palette("copper", len(unique_categories))  # Use a distinct color palette
# color_discrete = mcolors.ListedColormap(colors)
indianaPlot[2020] = pd.to_numeric(indianaPlot[2020], errors='coerce') # this is needed to make the legend_lwds work.it takes only float or numeric value whereas this was object.
# Plot the compname with the custom colormap
myax = indianaPlot.plot(column=2020, cmap="Greens",legend=True,legend_kwds={"label":"bu/ac","fmt": "{:.1f}", "orientation": "vertical", 'shrink':0.5, 'aspect':30})

# Overlay the indiana_counties_shape boundaries
indiana_counties_shape.plot(ax=myax, color="None", edgecolor="gray", alpha=0.4)

# Loop through each row in indianaPlot to add labels for both 'compname' and 'value'
for idx, row in indianaPlot.iterrows():
   if row['geometry'] is not None and not row['geometry'].is_empty:
        try:
            # Get the centroid of the geometry for placing the text
            centroid = row['geometry'].centroid
            # Get the compname and value
            compname = row['drainageclabbrvt']
            value = row[2020]
            # Place the text at the centroid of the polygon
            myax.text(centroid.x, centroid.y, f'{compname}', 
                      horizontalalignment='center', fontsize=8, color='black')
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
plt.xticks([])
plt.yticks([])
for spine in plt.gca().spines.values():
        spine.set_visible(False)
#plt.savefig(f"D:\OneDrive - purdue.edu\AutomatingAgFieldTrials\Fall2024\IndianaCountyPlots\indianaCountyWise_DCL_Yield2020.png", bbox_inches='tight', dpi=600)
plt.savefig(f"/Users/jhasneha/Documents/DOEjha/CodesResults/CountyIndiana/indianaCountyWise_DCL_Yield2020.png", bbox_inches='tight', dpi=600)

# Show the plot
plt.show()

#%%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from shapely.geometry import box
import pandas as pd
plt.style.use('dark_background')
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})

# Ensure the 'Value' column is numeric, and replace non-numeric values with NaN
indianaPlot[2020] = pd.to_numeric(indianaPlot[2020], errors='coerce')

# Drop rows with missing 'Value'
indianaPlot = indianaPlot.dropna(subset=[2020])

# Generate a color palette with exactly 7 colors (one for each drainage class) and assign them to the sorted classes
colors = sns.color_palette("copper", len(drainage_order))  # Generate the colors from light to dark
color_mapping = dict(zip(drainage_order, colors))  # Create a mapping of drainage class to color

# Now create a colormap using the sorted colors
color_list = [color_mapping[cls] for cls in drainage_order]
color_discrete = mcolors.ListedColormap(color_list)

# Ensure the values in 'drainageclabbrvt' are ordered to match the color order
indianaPlot['drainageclabbrvt'] = pd.Categorical(indianaPlot['drainageclabbrvt'], categories=drainage_order, ordered=True)
myax = indianaPlot.plot(column='drainageclabbrvt', cmap=color_discrete, legend=True, legend_kwds={"title": "DCL",'loc': 'center left', 'bbox_to_anchor': (1, 0.5)})

# Overlay the indiana_counties_shape boundaries
indiana_counties_shape.plot(ax=myax, color="None", edgecolor="gray", alpha=0.4)
plt.title(f"average yield in Indiana in 2020 was {round(indianaPlot[2020].mean(),1)} bu/ac")
# Loop through each row in indianaPlot to add bar-like rectangles for 'Value' at the centroid of each shape
for idx, row in indianaPlot.iterrows():
    if row['geometry'] is not None and not row['geometry'].is_empty:
        try:
            # Get the centroid of the geometry for placing the bar plot
            centroid = row['geometry'].centroid
            # Get the compname and value
            value = row[2020]
            # Place the text at the centroid of the polygon
            myax.text(centroid.x, centroid.y-10000, f'{value}', 
                      horizontalalignment='center', fontsize=6, color='black')
            # Adjust bar height based on value and scale the bar to the map
            bar_height = (value -indianaPlot[2020].mean())*500 # Adjust this scale factor for better visibility
            bar_width = 5  # Adjust width based on map scaling
            
            # Create a rectangle (box) geometry for the bar using shapely
            bar = box(centroid.x - bar_width / 2, centroid.y, centroid.x + bar_width / 2, centroid.y + bar_height)
            
            #baravg = box(centroid.x - bar_width / 2, centroid.y, centroid.x + bar_width / 2, centroid.y + bar_height)
            bar_color = 'green' if bar_height > 0 else 'red'

            # Plot the bar as a new geometry on the same axis
            plt.plot(*bar.exterior.xy, color=bar_color, alpha=0.7)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
plt.xticks([])
plt.yticks([])
for spine in plt.gca().spines.values():
        spine.set_visible(False)
#plt.savefig(f"D:\OneDrive - purdue.edu\AutomatingAgFieldTrials\Fall2024\IndianaCountyPlots\indianaCountywise_dclYield2020_barplot.png", bbox_inches='tight', dpi=600)
#plt.savefig(f"/Users/jhasneha/Documents/DOEjha/CodesResults/CountyIndiana/indianaCountywise_dclYield2020_barplot.png", bbox_inches='tight', dpi=600)

# Show the plot
plt.show()








#  # %%
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# plt.style.use('dark_background')
# plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})
# plt.rcParams.update({'font.size': 16})
# plt.rcParams.update({'legend.fontsize': 12})


# # Ensure that compname is a categorical column and ordered according to drainage_order
# indianaPlot["drainageclabbrvt"] = pd.Categorical(indianaPlot['drainageclabbrvt'], categories=drainage_order, ordered=True)

# indianaPlot = indianaPlot.dropna(subset=[2020])  # Drop rows where 'yield' is NaN
# indianaPlot = indianaPlot.dropna(subset=['drainageclabbrvt'])  # Drop rows where 'yield' is NaN

# plt.scatter(indianaPlot['drainageclabbrvt'], indianaPlot[2020], alpha=0.7, color="white", edgecolors="none")

# # Add labels and title
# plt.xlabel('Soil Component Name')
# plt.ylabel('county Yield (bu/ac)')
# plt.title('Component Name vs Corn Yield in 2020')

# # # Rotate x-axis labels if there are too many categories
# # plt.xticks(rotation=90)
# # Make y-axis ticks less dense
# ax = plt.gca()
# ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))  # nbins controls number of ticks
# for spine in plt.gca().spines.values():
#         spine.set_visible(False)

# plt.grid(color = 'azure', linestyle = '--', linewidth = 0.5)

# plt.tight_layout()
# #plt.savefig(f"D:\OneDrive - purdue.edu\AutomatingAgFieldTrials\Fall2024\IndianaCountyPlots\Scatterplot_indianaCountywise_dcl_yield2020.png", bbox_inches='tight', dpi=600)
# plt.savefig(f"/Users/jhasneha/Documents/DOEjha/CodesResults/CountyIndiana/Scatterplot_indianaCountywise_dcl_yield2020.png", bbox_inches='tight', dpi=600)

# plt.show()

# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'legend.fontsize': 12})


# Define the custom order for the x-ticks (compname)
drainage_order = ['ED', 'SED', 'WD', 'MWD', 'SPD', 'PD', 'VPD']

# Ensure that compname is a categorical column and ordered according to drainage_order
indianaPlot['drainageclabbrvt'] = pd.Categorical(indianaPlot['drainageclabbrvt'], categories=drainage_order, ordered=True)

# Drop rows where 'Value' or 'compname' are NaN
indianaPlot = indianaPlot.dropna(subset=[2020, 'drainageclabbrvt'])

# Sort the DataFrame according to the custom order of compname
indianaPlot = indianaPlot.sort_values('drainageclabbrvt')

# Debugging: Check the unique values in compname
print("Unique compname values in indianaPlot:", indianaPlot['drainageclabbrvt'].unique())

# Create scatter plot with compname on the x-axis and yield (Value) on the y-axis
plt.figure()
plt.scatter(indianaPlot['drainageclabbrvt'], indianaPlot[2020], alpha=1, color="beige", edgecolors="none")

# Add labels and title
plt.xlabel('Soil drainage class')
plt.ylabel('County Yield (bu/ac)')
plt.title('Drainage class vs Corn Yield in 2020')

# # Rotate x-axis labels if there are too many categories
# plt.xticks(rotation=90)

# Make y-axis ticks less dense
ax = plt.gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))  # nbins controls number of ticks
for spine in plt.gca().spines.values():
        spine.set_visible(False)
# Show the plot

plt.grid(color = 'azure', linestyle = '--', linewidth = 0.5)
plt.tight_layout()

#plt.savefig(f"D:\OneDrive - purdue.edu\AutomatingAgFieldTrials\Fall2024\IndianaCountyPlots\Scatterplot_indianaCountywise_dcl_yield2020.png", bbox_inches='tight', dpi=600)
plt.savefig(f"/Users/jhasneha/Documents/DOEjha/CodesResults/CountyIndiana/Scatterplot_indianaCountywise_dcl_yield2020.png", bbox_inches='tight', dpi=600)

plt.show()

# %%
mergedf=indianaPlot[["county","geometry","compname","drainageclabbrvt"]].copy()
IN_cornYield_pivot=IN_cornYield_pivot.merge(mergedf, left_on='county_name', right_on='county', how='left')
IN_cornYield_singlecolumn=IN_cornYield[["county_name","Value","year"]].copy()
IN_cornYield_singlecolumn=IN_cornYield_singlecolumn.merge(mergedf, left_on='county_name', right_on='county', how='left')
# %%

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.ticker import MaxNLocator
#plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'legend.fontsize': 12})

IN_cornYield_singlecolumn=IN_cornYield_singlecolumn.replace("None",math.nan)
IN_cornYield_singlecolumn["Value"] = pd.to_numeric(IN_cornYield_singlecolumn["Value"], errors='coerce')
IN_cornYield_singlecolumn= IN_cornYield_singlecolumn[IN_cornYield_singlecolumn["drainageclabbrvt"].notna()]
df_melted=IN_cornYield_singlecolumn.copy()
# Convert year column to numeric if necessary
df_melted['year'] = pd.to_numeric(df_melted['year'])

# Loop through each drainageclabbrvt and plot the corresponding data
for dcl in df_melted['drainageclabbrvt'].unique():
    plt.style.use('dark_background')
    
    # Filter data for the specific drainage class abbreviation
    df_filtered = df_melted[df_melted['drainageclabbrvt'] == dcl]
    
    # Plot the data, with each county having its own line
    sns.lineplot(x='year', y="Value", hue='county_name', data=df_filtered, marker='o', markersize=2, linewidth=1)
    sns.lineplot(x='year', y="Value", data=df_filtered, estimator='mean',errorbar="sd", color='white', linewidth=2, label='Mean')
    # Add title and labels
    plt.title(f'{dcl_dict_rev[dcl]}')
    plt.xlabel('Year')
    plt.ylabel('Corn Yield (bu/ac)')
    # Make y-axis ticks less dense
    myax = plt.gca()
    myax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))  # nbins controls number of ticks
    # Display the legend (with counties)
    #plt.legend(title='County', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.move_legend(myax, "upper left", bbox_to_anchor=(1, 1))
    # Show the plot for the current drainage class
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.grid(color = 'azure', linestyle = '--', linewidth = 0.5)
    plt.tight_layout()
    #plt.savefig(f"D:\OneDrive - purdue.edu\AutomatingAgFieldTrials\Fall2024\IndianaCountyPlots\Lineplot_indianaCountywise_dcl_yield2020_{dcl}.png", bbox_inches='tight', dpi=600)
    plt.savefig(f"/Users/jhasneha/Documents/DOEjha/CodesResults/CountyIndiana/Lineplot_indianaCountywise_dcl_yield2020_{dcl}.png", bbox_inches='tight', dpi=600)

    plt.show()






""""@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


Parent material kind

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""

# %%
# Sanitize input to remove any unwanted characters
def sanitize_input(user_input):
    return re.sub(r"[\'\"\\]", "", user_input)

# Function to fetch data for a given county code
def fetch_ssurgo_data_by_county(areasymbol):
    url = 'https://sdmdataaccess.nrcs.usda.gov/tabular/post.rest'

    # SQL query to fetch compname and muareaacres from the mupolygon and component tables
    # query = f"""SELECT TOP 5 mp.iacornsr , mu.muareaacres
    #             FROM mupolygon mu
    #             LEFT OUTER JOIN mapunit mp ON mu.mukey = mp.mukey
    #             WHERE mu.areasymbol = '{areasymbol}'
    #             """
    query = f"""SELECT TOP 2 cp.pmkind, SUM(mu.muareaacres) AS total_muareaacres
                FROM mupolygon mu
                LEFT OUTER JOIN component co ON mu.mukey = co.mukey
                LEFT OUTER JOIN copmgrp cpg ON co.cokey = cpg.cokey
                LEFT OUTER JOIN copm cp ON cpg.copmgrpkey = cp.copmgrpkey
                WHERE mu.areasymbol = '{areasymbol}'
                GROUP BY co.mukey, cp.pmkind
                ORDER BY total_muareaacres DESC
                """
    # URL-encode the query
    encoded_query = urllib.parse.quote_plus(query)

    # Prepare the data for the POST request
    data = f"query={encoded_query}&format=JSON"

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = None
    try:
        # Send the POST request
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Return the JSON data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred for {areasymbol}: {e}")
        return None

# Initialize an empty DataFrame to store results
Aggregate_County = pd.DataFrame(columns=["county", "compname","county_code", "total_muareaacres"])

# Loop through each county in the dictionary
for key, county in indiana_county_dict.items():
    # Format the county code to be like "IN001", "IN002", etc.
    county_code = f"IN{str(key).zfill(3)}"
    print("county_code: ",county_code)
    # Fetch data for the county
    ssurgo_data = fetch_ssurgo_data_by_county(county_code)
    print("ssurgo_data: ",ssurgo_data)
    # Check if data was returned
    if ssurgo_data and 'Table' in ssurgo_data:
        for row in ssurgo_data['Table']:
            # Add each row of data to the DataFrame
            print("row output: ",row)
            if row[0] is None or "None" in row[0]:
                #print("Nonetype encountered")
                continue
            else:
                Aggregate_County = pd.concat([Aggregate_County, pd.DataFrame([{"county": county,"county_code":county_code, "compname": row[0], "total_muareaacres": row[1]}])], ignore_index=True)
                #print("added")
                break #this statement is stopping the entry of multiple rows for each county.
print(Aggregate_County.head(2))


# %%
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'legend.fontsize': 10})

plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})
# Ensure the values in 'drainageclabbrvt' are ordered to match the color order
indianaPlot['compname'] = pd.Categorical(indianaPlot['compname'])

# Plot the data with the sorted colormap
myax = indianaPlot.plot(column='compname', cmap="Set2", legend=True, legend_kwds={"title": "Parent material",'loc': 'center left', 'bbox_to_anchor': (1, 0.5), 'frameon': False,'title_fontsize': 10})

# Overlay the Indiana county boundaries with transparency
indiana_counties_shape.plot(ax=myax, color="None", edgecolor="grey",linewidth=1, alpha=1)
for legend_handle in myax.get_legend().legend_handles:
    #print(legend_handle)
    legend_handle.set_markeredgewidth(0.3)
    legend_handle.set_markeredgecolor("black")

for spine in plt.gca().spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.savefig(f"D:\OneDrive - purdue.edu\AutomatingAgFieldTrials\Fall2024\IndianaCountyPlots\indiana_soil_pmkind.png", bbox_inches='tight', dpi=600)
#plt.savefig(f"/Users/jhasneha/Documents/DOEjha/CodesResults/CountyIndiana/indiana_soil_drainage_classes.png", bbox_inches='tight', dpi=600)

plt.show()


# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
plt.style.use('dark_background')
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'legend.fontsize': 10})
# # Get the unique categories from the 'compname' column
# unique_categories = indianaPlot['compname'].unique()
# colors = sns.color_palette("copper", len(unique_categories))  # Use a distinct color palette
# color_discrete = mcolors.ListedColormap(colors)
indianaPlot[2020] = pd.to_numeric(indianaPlot[2020], errors='coerce') # this is needed to make the legend_lwds work.it takes only float or numeric value whereas this was object.
# Plot the compname with the custom colormap
myax = indianaPlot.plot(column=2020, cmap="Greens",legend=True,legend_kwds={"label":"bu/ac","fmt": "{:.1f}", "orientation": "vertical", 'shrink':0.5, 'aspect':30})

# Overlay the indiana_counties_shape boundaries
indiana_counties_shape.plot(ax=myax, color="None", edgecolor="grey", alpha=0.4)

# Loop through each row in indianaPlot to add labels for both 'compname' and 'value'
for idx, row in indianaPlot.iterrows():
   if row['geometry'] is not None and not row['geometry'].is_empty:
        try:
            # Get the centroid of the geometry for placing the text
            centroid = row['geometry'].centroid
            # Get the compname and value
            compname = row['compname']
            value = row[2020]
            # Place the text at the centroid of the polygon
            myax.text(centroid.x, centroid.y, f'{compname}', 
                      horizontalalignment='center', fontsize=8, color='black')
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
plt.xticks([])
plt.yticks([])
for spine in plt.gca().spines.values():
        spine.set_visible(False)
plt.savefig(f"D:\OneDrive - purdue.edu\AutomatingAgFieldTrials\Fall2024\IndianaCountyPlots\indianaCountyWise_DCL_Yield2020.png", bbox_inches='tight', dpi=600)
#plt.savefig(f"/Users/jhasneha/Documents/DOEjha/CodesResults/CountyIndiana/indianaCountyWise_DCL_Yield2020.png", bbox_inches='tight', dpi=600)

# Show the plot
plt.show()
# %%
#%%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from shapely.geometry import box
import pandas as pd
plt.style.use('dark_background')
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'legend.fontsize': 10})

# Ensure the 'Value' column is numeric, and replace non-numeric values with NaN
indianaPlot[2020] = pd.to_numeric(indianaPlot[2020], errors='coerce')

# Drop rows with missing 'Value'
indianaPlot = indianaPlot.dropna(subset=[2020])

# Ensure the values in 'drainageclabbrvt' are ordered to match the color order
indianaPlot['compname'] = pd.Categorical(indianaPlot['compname'])
myax = indianaPlot.plot(column='compname', cmap="Set2", legend=True, legend_kwds={"title": "Parent Material",'loc': 'center left', 'bbox_to_anchor': (1, 0.5),'frameon': False,'title_fontsize': 10})

# Overlay the indiana_counties_shape boundaries
indiana_counties_shape.plot(ax=myax, color="None", edgecolor="gray", alpha=0.4)
plt.title(f"average yield in Indiana in 2020 \n was {round(indianaPlot[2020].mean(),1)} bu/ac", fontsize=12)
# Loop through each row in indianaPlot to add bar-like rectangles for 'Value' at the centroid of each shape
for idx, row in indianaPlot.iterrows():
    if row['geometry'] is not None and not row['geometry'].is_empty:
        try:
            # Get the centroid of the geometry for placing the bar plot
            centroid = row['geometry'].centroid
            # Get the compname and value
            value = row[2020]
            # Place the text at the centroid of the polygon
            myax.text(centroid.x, centroid.y-10000, f'{value}', 
                      horizontalalignment='center', fontsize=6, color='black')
            # Adjust bar height based on value and scale the bar to the map
            bar_height = (value -indianaPlot[2020].mean())*500 # Adjust this scale factor for better visibility
            bar_width = 5  # Adjust width based on map scaling
            
            # Create a rectangle (box) geometry for the bar using shapely
            bar = box(centroid.x - bar_width / 2, centroid.y, centroid.x + bar_width / 2, centroid.y + bar_height)
            
            #baravg = box(centroid.x - bar_width / 2, centroid.y, centroid.x + bar_width / 2, centroid.y + bar_height)
            bar_color = 'green' if bar_height > 0 else 'red'

            # Plot the bar as a new geometry on the same axis
            plt.plot(*bar.exterior.xy, color=bar_color, alpha=0.7)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
plt.xticks([])
plt.yticks([])
for spine in plt.gca().spines.values():
        spine.set_visible(False)
plt.savefig(f"D:\OneDrive - purdue.edu\AutomatingAgFieldTrials\Fall2024\IndianaCountyPlots\indianaCountywise_dclYield2020_barplot.png", bbox_inches='tight', dpi=600)
#plt.savefig(f"/Users/jhasneha/Documents/DOEjha/CodesResults/CountyIndiana/indianaCountywise_dclYield2020_barplot.png", bbox_inches='tight', dpi=600)

# Show the plot
plt.show()


#%%
# %%
mergedf=indianaPlot[["county","geometry","compname"]].copy()
IN_cornYield_pivot=IN_cornYield_pivot.merge(mergedf, left_on='county_name', right_on='county', how='left')
IN_cornYield_singlecolumn=IN_cornYield[["county_name","Value","year"]].copy()
IN_cornYield_singlecolumn=IN_cornYield_singlecolumn.merge(mergedf, left_on='county_name', right_on='county', how='left')
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.ticker import MaxNLocator
#plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'legend.fontsize': 10})

IN_cornYield_singlecolumn=IN_cornYield_singlecolumn.replace("None",math.nan)
IN_cornYield_singlecolumn["Value"] = pd.to_numeric(IN_cornYield_singlecolumn["Value"], errors='coerce')
IN_cornYield_singlecolumn= IN_cornYield_singlecolumn[IN_cornYield_singlecolumn["compname"].notna()]
df_melted=IN_cornYield_singlecolumn.copy()
# Convert year column to numeric if necessary
df_melted['year'] = pd.to_numeric(df_melted['year'])

# Loop through each drainageclabbrvt and plot the corresponding data
for component in df_melted['compname'].unique():
    plt.style.use('dark_background')
    
    # Filter data for the specific drainage class abbreviation
    df_filtered = df_melted[df_melted['compname'] == component]
    
    # Plot the data, with each county having its own line
    sns.lineplot(x='year', y="Value", hue='county_name', data=df_filtered, marker='o', markersize=2, linewidth=1)
    sns.lineplot(x='year', y="Value", data=df_melted, estimator='mean',errorbar="sd", color='white', linewidth=2, label='stateMean',zorder=40)

    # Add title and labels
    plt.title(f'{component}')
    plt.xlabel('Year')
    plt.ylabel('Corn Yield (bu/ac)')
    # Make y-axis ticks less dense
    myax = plt.gca()
    myax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))  # nbins controls number of ticks
    # Display the legend (with counties)
    #plt.legend(title='County', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.move_legend(myax, "upper left", bbox_to_anchor=(1, 1))
    # Show the plot for the current drainage class
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.grid(color = 'azure', linestyle = '--', linewidth = 0.5)
    plt.tight_layout()
    plt.savefig(f"D:\OneDrive - purdue.edu\AutomatingAgFieldTrials\Fall2024\IndianaCountyPlots\Lineplot_indianaCountywise_dcl_yield2020_{dcl}.png", bbox_inches='tight', dpi=600)
    #plt.savefig(f"/Users/jhasneha/Documents/DOEjha/CodesResults/CountyIndiana/Lineplot_indianaCountywise_dcl_yield2020_{dcl}.png", bbox_inches='tight', dpi=600)

    plt.show()


# %%
