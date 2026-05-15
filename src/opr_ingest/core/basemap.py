"""Cartopy/GeoViews basemaps."""

import cartopy.crs as ccrs
import geoviews.feature as gf


def create_antarctica_basemap():
    """Antarctic ocean + coastline basemap in EPSG:3031."""
    epsg_3031 = ccrs.Stereographic(central_latitude=-90, true_scale_latitude=-71)
    return (gf.ocean.options(scale='50m').opts(projection=epsg_3031)
            * gf.coastline.options(scale='50m').opts(projection=epsg_3031))
