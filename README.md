# Meme Classifier

There are so many templates for the internet meme images.

This tool helps to automatically determine which template an image uses.


## How to use it

First, you have to put all the meme templates in a subdirectory.

    from memeclass import MemeClassifier
    
    sub_dir_name = 'templates'
    img_file_name = 'aQq9GqW.jpg'
    
    mc = MemeClassifier(sub_dir_name)
    meme_name = mc.classify(img_file_name)
    print img_file_name, meme_name

About 60 templates are included in the tool by default.

You can add your own templates!

## Dependency

You have to install the Python Opencv:

    sudo apt-get install python-opencv

## Method

This tool use basic digital image process knowledge:

* Spatial filtering
* Histogram intersection
* Hue extraction
