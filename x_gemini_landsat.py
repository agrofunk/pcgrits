
#Create an AWS account and IAM user with appropriate permissions to access the Landsat STAC catalog. Alternatively, use existing credentials with sufficient access.
#Set up environment variables to contain your AWS credentials securely:
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="YOUR_REGION"  # Replace with your region (e.g., "us-east-1")

import stackstac
import xarray as xr
import boto3

client = stackstac.Client(
    base_url="https://landsat-pds.s3.amazonaws.com/stac",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_DEFAULT_REGION")
)

collection = "LC08"
bbox = (-46.5, -23.5, -45.5, -22.5)  # Example bounding box for São Carlos, Brazil
datetime = "2023-01-01/2023-12-31"
bands = ["B4", "B5", "B7"]

query = {
    "collection": collection,
    "bbox": bbox,
    "datetime": datetime,
    "ids": None  # Optional: Filter by specific item IDs
}

datacube = client.get_datacube(query=query)


# Iterate through datacube items and download data:
datasets = []

for item in datacube.items():
    file_id = item.id
    print(f"Processing item: {file_id}")

    # Download data using StackSTAC's efficient methods
    data_urls = item.get_asset_urls("data")
    for band in bands:
        url = data_urls[band]
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raise an exception for non-200 status codes
            dataset = xr.open_dataset(response.raw, engine="rasterio")
            dataset.name = band  # Assign band name for clarity

            # Apply custom preprocessing if needed
            # ...

            datasets.append(dataset)

# Optionally, combine datasets into a single multi-band xarray Dataset:
multiband_dataset = xr.combine_nested(datasets, concat="bands")

output_path = "landsat_datacube.nc"
multiband_dataset.to_netcdf(  XXXX  )
