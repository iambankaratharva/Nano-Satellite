#necessary imports
import json
import cv2
import os
import numpy as np

#loading the json file
with open('train/sample.json') as file:
    data = json.load(file)
    
#creating the ground_truth folder if it doesn't exist
os.makedirs('ground_truth', exist_ok=True)

#parsing through each image description in json    
for image in data.keys():
    #data and metadata about each image
    temp_data = data[image]
    
    #loading the image to capture its dimensions
    image = cv2.imread('train/'+temp_data['filename'])
    
    #creating a new ground truth image
    mask = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    
    #appending the mask for the regions in the image
    for region in temp_data['regions']:
        coord = [[x, y] for x, y in zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y'])] 
        coord = np.array(coord, np.int32)
        coord = coord.reshape((-1, 1, 2))
        mask = cv2.fillPoly(mask, [coord], 255)
    
    #saving the ground truth masks
    cv2.imwrite('ground_truth/'+temp_data['filename'], mask)