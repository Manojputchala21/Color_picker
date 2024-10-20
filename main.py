import cv2
import httpx
import numpy as np
from pydantic import BaseModel
from sklearn.cluster import KMeans
from fastapi import FastAPI
import requests
import  uvicorn
from starlette.responses import HTMLResponse


class ImageData(BaseModel):
    image_url: str
    color_hex_code: str
    n_colors: int

# Convert hex code to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

app = FastAPI()
# http://127.0.0.1:8000/getClosestColorCode?image_url=https://starappsstudio.s3.amazonaws.com/v2/apps/vsk_dev/sas-manoj/groups/2/300/gloves.webp&color_hex_code="800020"&n_colors=5
@app.get("/getClosestColorCode/")
async def getClosestColorCode(image_url: str, color_hex_code: str, n_colors: int):
    # Read the image
    response = requests.get(image_url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # image = cv2.imread(image_data.image_url)
    print(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)

    dominant_colors = kmeans.cluster_centers_.astype(int)

    static_color_rgb = hex_to_rgb(f"#{color_hex_code}")
    min_distance = float('inf')
    closest_color = None
    for color in dominant_colors:
        distance = np.linalg.norm(color - static_color_rgb)
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    hex_value = '#{:02x}{:02x}{:02x}'.format(closest_color[0], closest_color[1], closest_color[2])

    return HTMLResponse(f"<div style=width:200px;height:150px;background-color:{hex_value}></div><h1 style=color:{hex_value}>{hex_value}</h1>")

# image_url = "/Users/manoj/Downloads/burgundy.jpg"
# color_hex_code = "#800020"  # Replace with your static hex color
# closest_color = getClosestColorCode(image_url, color_hex_code)
# print(f"Closest color in the image: {closest_color}")
