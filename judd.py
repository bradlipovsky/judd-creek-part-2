import os
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def king_county_csv_loader(data_dir):
    dfs = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv") and filename.startswith("Hydrology"):
            print(filename)
            filepath = os.path.join(data_dir, filename)

            with open(filepath) as f:
                lines = [line.rstrip(',\n') for line in f]

            # Split all lines into lists of values
            split_lines = [line.split(',') for line in lines]
            raw_header = split_lines[0]

            # Drop the last column from the header (it's just a description)
            raw_header = raw_header[:-1]

            # Identify known fields present in this file
            known_fields = ['Precipitation (inches)', 'Stage (ft)', 'Discharge (cfs)']
            data_fields = [col for col in raw_header if col in known_fields]

            # Base header (Site_Code, Date)
            header = raw_header[:2] + data_fields + ['Flag1', 'Flag2', 'Flag3']
            max_cols = len(header)

            data_lines = split_lines[1:]

            # Pad or trim rows to match max_cols
            padded_rows = [row[:max_cols] + [''] * (max_cols - len(row)) for row in data_lines]

            # Create dataframe
            df = pd.DataFrame(padded_rows, columns=header)
            dfs.append(df)
            print(f"✔ Loaded {filename} ({len(df)} rows)")

    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicate columns by keeping the first occurrence
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Convert date column to datetime
#     if 'Collect Date (local)' in merged_df.columns:
#         merged_df['Collect Date (local)'] = pd.to_datetime(
#             merged_df['Collect Date (local)'], format="%m/%Y", errors='coerce'
#         )
    if 'Collect Date (local)' in merged_df.columns:
        # First try full month/day/year
        dates = pd.to_datetime(merged_df['Collect Date (local)'], format="%m/%d/%Y", errors='coerce')

        # Then fill in missing ones with month/year format
        mask = dates.isna()
        dates[mask] = pd.to_datetime(merged_df.loc[mask, 'Collect Date (local)'], format="%m/%Y", errors='coerce')

        # Assign back to DataFrame
        merged_df['Collect Date (local)'] = dates


    # Convert numeric columns if they exist
    expected_cols = ['Precipitation (inches)', 'Stage (ft)', 'Discharge (cfs)']
    for col in expected_cols:
        matched_cols = [c for c in merged_df.columns if c.strip() == col]
        if matched_cols:
            actual_col = matched_cols[0]
            merged_df[actual_col] = pd.to_numeric(merged_df[actual_col], errors='coerce')

    return merged_df



import numpy as np
from shapely.geometry import Polygon, LineString
from scipy.spatial import Voronoi
import geopandas as gpd

def make_voronoi_gdf(merged_2019, geojson_path="data/map.geojson"):
    """
    Build a clipped Voronoi GeoDataFrame from site locations and precipitation data.
    
    Parameters:
        merged_2019: pd.DataFrame with columns ['x', 'y', 'Site_Code', 'Avg Precip (inches)']
        geojson_path: path to boundary GeoJSON file (default: 'data/map.geojson')
    
    Returns:
        GeoDataFrame with Voronoi polygons clipped to the boundary and area columns.
    """
    from shapely.geometry import Polygon
    from scipy.spatial import Voronoi

    # Extract inputs
    points = merged_2019[['x', 'y']].values
    site_codes = merged_2019['Site_Code'].values
    precips = merged_2019['Avg Precip (inches)'].values

    # Voronoi computation
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # Load and project boundary
    boundary = gpd.read_file(geojson_path).to_crs('EPSG:2926')
    boundary_poly = boundary.unary_union

    # Clip and store polygons
    polygons, codes, vals = [], [], []
    for i, region in enumerate(regions):
        poly = Polygon(vertices[region])
        if not poly.is_valid:
            continue
        clipped = poly.intersection(boundary_poly)
        if not clipped.is_empty:
            polygons.append(clipped)
            codes.append(site_codes[i])
            vals.append(precips[i])

    # Build GeoDataFrame
    gdf_voronoi = gpd.GeoDataFrame({
        'Site_Code': codes,
        'Avg Precip (inches)': vals,
        'geometry': polygons
    }, crs='EPSG:2926')

    # Compute area
    gdf_voronoi['Area (m²)'] = gdf_voronoi.geometry.area
    gdf_voronoi['Area (km²)'] = gdf_voronoi['Area (m²)'] / 1e6

    return gdf_voronoi

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite polygons.
    Source: https://gist.github.com/pv/8036995
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 100  # Big enough to enclose outer areas

    # Construct a map from ridge points to ridge vertices
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue

        # Infinite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue  # finite ridge

            # Compute the missing endpoint
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # Reorder region's vertices
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)
