**Google Colab image segmentation project with Matterport Mask-RCNN implementation  on Keras and TensorFlow 2.14.0 and Python 3.10.12**


**Step 1: Collect web images from Google Images**
To copy images, I used a tool that simulates a person manually downloading images from Google (Scraping.py) . Selenium is an open-source web automation tool that can perform tasks by connecting Python to a web browser. Additionally, I needed to download a web driver for the corresponding version of Google Chrome (version 129).

**Step 2:Image Annotation**

The images were annotated using the VGG Image Annotator (VIA) tool. This tool allows for the annotation of objects in images, where each object is assigned a mask that describes its shape and location in the image. Each object, such as an item of clothing or an accessory, is manually outlined and labeled with the appropriate class.

![Annotation](https://github.com/AnniRanok/Segmentation/blob/main/IMG_6902.jpg)

**Saving Annotation Results:**

After completing the annotation, the data from VIA is exported in JSON/CSV format. Each object in the image has mask coordinates (in RLE format or a list of pixels), the object's class, and the image dimensions.  

Mask_Creating.ipynb

**Preparation of the image_df file:**

After obtaining the annotation results, all data is converted into a table, which is stored as a DataFrame (using the Pandas library). The main columns of this DataFrame are:

**ImageId:** A unique identifier for the image (without the file extension), used to link each object to the corresponding image.
**Height:** The height of the image (in pixels), to know the size of the mask that corresponds to this image.
**Width:** The width of the image (in pixels), similarly used for scaling the masks.
**EncodedPixels:** The masks for each object, represented in Run-Length Encoding (RLE) format. This is a compressed format that encodes the number of background and object pixels.
**ClassId:** The identifier of the object class, which represents the type of object (e.g., jacket, dress, socks, etc.).

**Step 3:Configure the project**

I used a modified version of the popular Mask-RCNN implementation from Matterport (https://github.com/matterport/Mask_RCNN). While the original version is restricted to older TensorFlow and Keras versions, I opted for z-mahmud22's adaptation (https://github.com/z-mahmud22/Mask-RCNN_TF2.14.0) that supports TensorFlow 2.14+ with integrated Keras, enabling me to leverage the latest advancements in deep learning.  

For our custom dataset, we need to add new classes (label_names) and override the configuration in the parent class it inherits from. To see the values of hyperparameters in the base configuration (e.g., LEARNING_RATE, STEPS_PER_EPOCH, etc.), we'll refer to the parent class in the Config.py script. The specific configuration is defined in the FashionConfig class. In addition to updating the configuration for the dataset, we define a custom class, FashionDataset, where we also add some functions to the existing Dataset class in the utils module.  

The FashionDataset class, which inherits from the utils.Dataset class (in model.py).It is designed to prepare data for training a model based on clothing segmentation.The Dataset class in the Mask R-CNN library serves as a base class for creating specific datasets used to train the model.
Using the utils.Dataset base class template ensures that FashionDataset will be compatible with other pieces of code that expect Dataset objects,
thus preserving the common structure and behavior.  

To train our model, we first load the custom FashionDataset class, which includes the necessary configuration and functions for data preparation. The prepare function (train_dataset = FashionDataset(train_df) and train_dataset.prepare() ) will handle loading and preprocessing both the training and validation datasets. Subsequently, we'll instantiate the model using the FashionConfig, which contains the specific hyperparameters for our task.
There are callbacks to silence for TensorBoard and save checkpoints for each epoch in the log folder.  

Data training using the configuration of the Mask R-CNN model and pre-prepared weights mask_rcnn_coco.h5 that we load into the model. The model should be trained in two processes. The first is "header learning". This is a mask label training class to predict the mask label of each object.We limit ourselves to training only the last layers of the network, which are responsible for classification and segmentation (for example, the classification layer, boxes and masks). This layer depends more on the specifics of our data (for example: categories of clothing items).  

Augmentation can help make training more robust to variations in the data. We load the weights model_weights.h5 of the model, then the model is trained on all layers (layers='all') for 16 epochs using augmentations.

Each epoch in the training process exhibits variations in loss values. Given that Mask-RCNN incorporates an Region Proposal Network (RPN) which predicts region proposals along with class labels, bounding boxes, and mask predictions, we observe five distinct types of losses to be minimized. For a detailed breakdown of these losses, please refer to the model.py script.

* rpn_class_loss = RPN anchor classification loss
* rpn_bbox_loss = RPN bounding box regression loss
* mrcnn_class_loss = Mask R-CNN classifier head loss
* mrcnn_bbox_loss = Mask R-CNN bounding box refinement loss
* mrcnn_mask_loss = Mask binary cross-entropy loss for mask head  

To generate predictions, we load the model weights corresponding to the epoch with the lowest validation loss from the saved checkpoints in the logs directory. The predicted results are then visualized. The first element of the results array (r=r[0]) is a dictionary containing the following key-value pairs:  

* **rois:** A list of bounding boxes representing the regions of interest for detected objects.
* **masks:** A set of masks, each corresponding to a detected object, indicating the precise pixel-level segmentation.
* **class_ids:** Integer identifiers for the predicted class of each object.
* **scores:** Confidence scores associated with each predicted class, indicating the model's certainty about the classification.  

****____________________________________________________________________________________________________________****





