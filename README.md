The data set can be downloaded from https://bbbc.broadinstitute.org/BBBC006. Only the necessary sets of z-stack levels 
can be downloaded. Before going to the next step, make sure to extract the images from the archives. 

The images from each z-stack level must be inside a directory named `z` and the digit/number representing the level 
(i.e. `z0` for images with z-stack 0). All these directories must be contained within a directory named `all`. An 
example is shown in the tree-structure below.

To process the images for usage and split them in train-validation-test sets, some configuration settings have to be
made. Go to the `config.py` file in the `models` directory and set the `TEST_RATIO` and `VALIDATION_RATIO` variables.
The remaining ratio of images will be used for the training set. Also set the `STACKS` variable accordingly. It 
describes which z-stack levels will be used throughout the program. In the `prepare_data.py` script, set the 
`triplets_train` variable accordingly. 

To prepare the train-val-test sets, run the following command:  `python prepare_data.py`. Three `train`, `test`, and 
`val` directories will be created, with the hierarchy shown in the tree-structure below.

```.
+-- dataset
|   +-- dataset.py
|   +-- prepare_data.py
|   +-- all
|   |   +-- z0
|   |   +-- z1
|   |   ....
|   |   +-- z33
|   +-- test
|   |   +-- raw
|   |   |   +-- z0
|   |   |   ....
|   |   |   +-- z33
|   |   +-- processed
|   |   |   +-- z0
|   |   |   ....
|   |   |   +-- z33
|   +-- train
|   |   +-- raw
|   |   +-- processed
|   +-- val
|   |   +-- raw
|   |   +-- processed
+-- models
|   +-- metrics.py
|   +-- model.py
|   +-- test.py
|   +-- utils.py
|   +-- visualizations.py
+-- requirements.txt
```

Other configurations can be set from the `config.py` file. Modify them accordingly before running the models.

To train a model, use the command `python model.py -experiment="some_name" --slide-type="some_type" 
--model-type="some_type"`. Replace the contents of the quotes accordingly. Also, set the `LAYERS_DEPTH` variable in the
`config.py` file accordingly.

To test a model, use the command `pytohn test.py -experiment="some_name --slide-type="some_type" [-vnum=0 -epoch=0 
step=0]`. Replace the arguments accordingly. The last 3 arguments are optional and must be provided only when an 
experiment has not been tested before. Check `test.py` for more information on how the code can be used to test various
properties of the models / generate visualizations.


