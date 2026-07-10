import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import os

geojson_path = os.path.join(os.path.dirname(__file__), "geoml_hackerthon.geojson")
output_path = os.path.join(os.path.dirname(__file__), "real_satellite_image.png")

gdf = gpd.read_file(geojson_path)
gdf = gdf[gdf.geometry.geom_type == 'Polygon']
total_bounds = gdf.total_bounds
min_lon, min_lat, max_lon, max_lat = total_bounds

print(f"Fetching satellite image for: {min_lon}, {min_lat} -> {max_lon}, {max_lat}")

fig, ax = plt.subplots(figsize=(10, 10))
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=18, crs='epsg:4326')
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
plt.axis('off')
plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
plt.close()

print(f"Saved real satellite image to: {output_path}")
