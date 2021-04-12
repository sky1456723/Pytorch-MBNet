# Pytorch-MBNet
A pytorch implementation of MBNET: MOS PREDICTION FOR SYNTHESIZED SPEECH WITH MEAN-BIAS NETWORK

## Training
To train a new model, please run ```train.py```, the input arguments are:

* --data_path: The path of the directory containing all .wav files of VCC-2018 and 
the train/dev/test split files (the files in  ```/data```).
* --save_dir: The path of the directory to save the trained models. Please create the directory before training.
* --total_steps: The total #training step in the training.
* --valid_steps: Do the validation every #(valid_steps) of training update.
* --log_steps: Log the tensorboard every #(log_steps) of training update.
* --update_freq: Gradient accumulation, the default value is 1 (no accumulation).

## Testing
To test on VCC-2018, please run ```test.py```, the input arguments are:
* --model_path: The path to the saved model.
* --idtable_path: The path to the "judge id-number" mapping table file used during training.
* --step: The time step for tensorboard log, which can be the same as the training steps.
* --split: The valid/test split of data to be used in the testing.

## Inference
After training on the VCC data, the model can be utilized to inference on other data. The input arguments are ```--data_path, --model_path, --save_dir```, which are similar to the above. Notice that the bias-net is not used since in this code the ground-truth judge ids are assumed to be unavailable.

The pre-trained model can be found in ```pre_trained```.

