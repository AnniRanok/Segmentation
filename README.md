**Google Colab image segmentation project with Matterport Mask-RCNN implementation on Tensorflow 2.14**


Step 1: Collect web images from Google Images





**Image Annotation:**
The images were annotated using the VGG Image Annotator (VIA) tool. This tool allows for the annotation of objects in images, where each object is assigned a mask that describes its shape and location in the image. Each object, such as an item of clothing or an accessory, is manually outlined and labeled with the appropriate class.

**Saving Annotation Results:**

After completing the annotation, the data from VIA is exported in CSV format. Each object in the image has mask coordinates (in RLE format or a list of pixels), the object's class, and the image dimensions.

**Preparation of the image_df file:**

After obtaining the annotation results, all data is converted into a table, which is stored as a DataFrame (using the Pandas library). The main columns of this DataFrame are:

**ImageId:** A unique identifier for the image (without the file extension), used to link each object to the corresponding image.
**Height:** The height of the image (in pixels), to know the size of the mask that corresponds to this image.
**Width:** The width of the image (in pixels), similarly used for scaling the masks.
**EncodedPixels:** The masks for each object, represented in Run-Length Encoding (RLE) format. This is a compressed format that encodes the number of background and object pixels.
**ClassId:** The identifier of the object class, which represents the type of object (e.g., jacket, dress, socks, etc.).