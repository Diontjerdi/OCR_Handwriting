### Required

To install required packages:

    pip install -r requirements.txt


### Running scripts

Run scripts from main directory to keep relative paths working, e.g.

    python tools\create_labels.py


### Download pretrained model

# TO DO: add link to model?

Download my pretrained model, or train your own, and put it in:

    $models\


### Test demo

    python demo.py PATH_TO_IMG IMG_TYPE

Examples:

    python demo.py test_char.png char
    python demo.py test_doc.png doc


### Train

Give path to configuration file as argument for training.
	
	python cnn\train.py conf\train.cfg


#### Data

Data should have this basic structure:

    $data\test\*.png      # test images
    $data\train\*.png     # train images
    $data\test.txt        # csv list with test image paths and labels
    $data\train.txt       # csv list with train image paths and labels


### Pydoc

    python tools\pydoc.py -w PATH_TO_FILE  # Include '.py' in filename

