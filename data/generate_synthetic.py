"""Generate synthetic satellite image + mask from GeoJSON polygons (fallback)."""
import geopandas as gpd
from PIL import Image, ImageDraw
import os

GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "geoml_hackerthon.geojson")
OUTPUT_IMG = os.path.join(os.path.dirname(__file__), "synthetic_satellite.png")
OUTPUT_MASK = os.path.join(os.path.dirname(__file__), "synthetic_satellite_mask.png")

def generate():
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf = gdf[gdf.geometry.geom_type == "Polygon"]
    gdf = gdf.drop_duplicates(subset=["id"])
    total_bounds = gdf.total_bounds
    min_lon, min_lat, max_lon, max_lat = total_bounds

    w, h = 512, 512
    img = Image.new("RGB", (w, h), color=(20, 20, 20))
    draw_img = ImageDraw.Draw(img)
    mask = Image.new("L", (w, h), 0)
    draw_mask = ImageDraw.Draw(mask)

    for _, row in gdf.iterrows():
        coords = []
        for lon, lat in row.geometry.exterior.coords:
            x = int((lon - min_lon) / (max_lon - min_lon) * (w - 1))
            y = int((max_lat - lat) / (max_lat - min_lat) * (h - 1))
            coords.append((max(0, min(w-1, x)), max(0, min(h-1, y))))
        draw_img.polygon(coords, fill=(255, 0, 0), outline=(255, 255, 255))
        draw_mask.polygon(coords, fill=1)

    img.save(OUTPUT_IMG)
    mask.save(OUTPUT_MASK)
    print(f"✅ Synthetic image: {OUTPUT_IMG}")
    print(f"✅ Synthetic mask: {OUTPUT_MASK}")

if __name__ == "__main__":
    generate()
