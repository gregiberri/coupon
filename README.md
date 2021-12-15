# Traffic light recognition
### Requirements
The required packages can be found in *config/env_files/coupon_env.yml*. 
Dependencies could be installed by running:
> conda env create -f config/env_files/coupon_env.yml

### Configuration
The experiments are run according to configurations. The config files for those can be found in 
*config/config_files*.
Configurations can be based on each other. This way the code will use the parameters of the specified 
base config and only the newly specified parameters will be overwritten.
 
The base config file is *base.yaml*. A hpo example can be found in *base_hpo.yaml*
which is based on *base.yaml* and does hyperparameter optimization only on the specified parameters.
An example for test based on *base.yaml* can be found in *test.yaml*.

### Arguments
The code should be run with arguments: 

--id_tag specifies the name under the config where the results will be saved \
--config specifies the config name to use (eg. config "base" for *config/config_files/base.yaml*)\
--mode can be 'train', 'val', 'test' or 'hpo'

### Required data
The required data's path should be specified inside the config file like:
> data: \
  &emsp; params: \
  &emsp; dataset_path: 'dataset' \
  &emsp; filename: 'coupon_data' \

During train and hpo the raw data should be under *{dataset_path}/{filename}*, 
or there should be a train and val csv under dataset_path\
During test the file should be under *{dataset_path}/{filename}*.  

### Saving and loading experiment
The save folder for the experiment outputs can be set in the config file like:
> id: "base"\
  env: \
  &emsp; result_dir: 'results'

All the experiment will be saved under the given results dir: {result_dir}/{config_id}/{id_tag arg}
1. (feature importance png if decision trees are used)
2. accuracy txt
3. the best model
4. (val/test preds csv if it is set to be saved)

If the result dir already exists and contains a model file then the experiment will automatically resume
(either resume the training or use the trained model for inference.)

### Usage
##### Training
To train the model use. During training besides the best model, the feature inmortances are saved 
(id the used model is a decision tree):
> python run.py --config base --mode train

#### Eval
For eval the  results dir ({result_dir}/{config_id}/{id_tag arg}) should contain a model as 
*model_best.pth.tar* or *model.pkl*. During eval the validation file will be inferenced and the accuracy will be calculated.
> python run.py --config base --mode val

#### Test
For test the  results dir ({result_dir}/{config_id}/{id_tag arg}) should contain a model as 
*model_best.pth.tar* or *model.pkl*. During test the predictions will be saved.\
A pretrained model can be found in [here](https://drive.google.com/file/d/1sILKQ3MkCy7lSEOz8E7s3u5G5bJPpVzs/view?usp=sharing). 
For simplicity it is recommended to copy it under *results/base/base* and just change the dataset path to yours in *config/config_files/test.yaml*.
> python run.py --config test --mode test

#### HPO
For hpo use:
> python run.py --config base_hpo --mode hpo

### Example of the results:
HPO result:
![Screenshot from 2021-11-16 01-07-55](https://user-images.githubusercontent.com/36601982/141872068-600b5cea-11d9-44d2-b84a-7566fa5df99c.png)

Confusion matrix:
![confusion_matrix_val_40](https://user-images.githubusercontent.com/36601982/141871724-630f07af-82db-4c28-92f1-ca6d77bcba20.png)

By class F1 score:
![F1_val_40](https://user-images.githubusercontent.com/36601982/141871759-d975873a-a07f-4cb0-b2d3-f566982b461f.png)
