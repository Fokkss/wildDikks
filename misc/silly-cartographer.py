# I don't think this small script is going to get used alot, so I decided that it would be too much effort to make it more "tolerable" than it is now.
# It converts geodetic WGS84 to cartesian centered at the first entry.
# The utilized tangent model should be fine for our case since all the points are contatined inside a 2.5x10 km rectangle.
import os
import pandas as pd
from pyproj import Transformer

def convert_gps_to_local_cartesian(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    lats = df['latitude'].values
    lons = df['longitude'].values
    
    lat0, lon0 = lats[0], lons[0]
    
    pipeline_string = (
        f"+proj=pipeline "
        f"+step +proj=axisswap +order=2,1 "
        f"+step +proj=latlong +ellps=WGS84 "
        f"+step +proj=cart +ellps=WGS84 "
        f"+step +proj=topocentric +ellps=WGS84 +lat_0={lat0} +lon_0={lon0}"
    )
    
    transformer = Transformer.from_pipeline(pipeline_string)
    
    alts = [0.0] * len(lats)
    x_coords, y_coords, _ = transformer.transform(lons, lats, alts)
    
    df['X_meters'] = x_coords
    df['Y_meters'] = y_coords
    
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} points. With origin at Lat: {lat0}, Lon: {lon0}")

if __name__ == "__main__":
    input_file = input("Enter the path to input CSV file: ").strip()
    output_filename = input("Enter the name for the output CSV file (e.g., cartesian.csv): ").strip()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, output_filename)
    
    if os.path.exists(input_file):
        convert_gps_to_local_cartesian(input_file, output_file)
    else:
        print(f"Error: '{input_file}' does not exist.")

