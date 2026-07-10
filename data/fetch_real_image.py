"""
Fetch a real satellite image matching the GeoJSON bounds.
Uses Esri World Imagery via contextily (no API key required).
"""
import geopandas as gpd
import contextily as ctx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "geoml_hackerthon.geojson")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "real_satellite_image.png")

def fetch():
    print("📖 Loading GeoJSON...")
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf = gdf[gdf.geometry.geom_type == "Polygon"]
    total_bounds = gdf.total_bounds
    min_lon, min_lat, max_lon, max_lat = total_bounds

    print(f"🌍 Bounds: {min_lon:.4f}, {min_lat:.4f} -> {max_lon:.4f}, {max_lat:.4f}")
    print("⏳ Downloading satellite tile from Esri World Imagery...")

    fig, ax = plt.subplots(figsize=(10, 10))
    ctx.add_basemap(
        ax,
        source=ctx.providers.Esri.WorldImagery,
        zoom=17,
        crs="epsg:4326",
    )
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.axis("off")

    plt.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()

    print(f"✅ Real satellite image saved to: {OUTPUT_PATH}")
    print(f"   File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

if __name__ == "__main__":
    fetch()
