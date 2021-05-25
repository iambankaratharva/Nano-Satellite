#necessary imports
import xml.etree.ElementTree as ET
import os

classes = ['Fire']

os.makedirs('YOLOv5', exist_ok=True)

#going through each PASCAL_VOC annotation
for annotation in os.listdir('PASCAL VOC XMLs'):
    #removing unnecessary files
    if annotation=='temp':
        continue
    
    #opening each PASCAL VOC XML file
    with open(f'PASCAL VOC XMLs/{annotation}') as file:
        tree = ET.parse(file)
   
    #parsing through the root
    root = tree.getroot()
    
    #capturing metadata about image
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    #going through each object in the image
    output = ''
    for obj in root.iter('object'):
        #capturing class information
        cls = obj.find('name').text
        cls_id = classes.index(cls)
        
        #capturing the bounding box
        bnd_box = obj.find('bndbox')
        bnd_box = (float(bnd_box.find('xmin').text),
                   float(bnd_box.find('xmax').text),
                   float(bnd_box.find('ymin').text),
                   float(bnd_box.find('ymax').text))
        
        #converting to the YOLOv5 required format
        dw = 1./width
        dh = 1./height
        x = (bnd_box[0]+bnd_box[1])/2.0-1
        y = (bnd_box[2]+bnd_box[3])/2.0-1
        w = bnd_box[1]-bnd_box[0]
        h = bnd_box[3]-bnd_box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        
        #appending the converted data to output string
        output = output+str(cls_id)+' '+' '.join([str(a) for a in (x, y, w, h)])+'\n'
    
    #writing the data in a new file
    with open(f"YOLOv5/{annotation[:-4]}.txt", 'w') as file:
        file.write(output)
