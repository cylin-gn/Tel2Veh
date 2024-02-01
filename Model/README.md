
# Experimental Code for Spatio-Temporal Fusion Framework
## for integrating GCT and vehicle flow to predict the future vehicle flows in camera-free areas.

![dataset](https://github.com/cylin-gn/Tel2Veh/blob/main/Figure/fusion_model.png)

This is a Pytorch implementation of the proposed Spatio-Temporal Fusion Framework.

## Datasets for training
The **processed train/val/test data structures file** is available, 
- data structures file for GCT flow: [Here](https://github.com/cylin-gn/Tel2Veh/tree/main/Data/hsin_9_CCTV_0600_1900_rename)
- data structures file for Vehiclw flow: [Here](https://github.com/cylin-gn/Tel2Veh/tree/main/Data/hsin_9_CCTV_0600_1900_rename)

## Models

### Here, we employ [Graph Wavenet (GWNET)](https://github.com/nnzhan/Graph-WaveNet) as an example of an STGNN integrated into the framework.

Download and store the trained models in 'pretrained' folder as follow:

```
./Model/save
```
Here is the saved model of GWNET: the first two are pre-trained for feature extraction, while the last one is trained for predicting vehicle flow in stage 2.
- Pretrained for GCT Flow Feature Extraction (STGNN-1): [exp202312151449_0.pth](https://github.com/cylin-gn/Tel2Veh/blob/main/Model/save/exp202312151449_0.pth)
- Pretrained for Vehicle Flow Feature Extraction (STGNN-2):" [exp202301041709_0.pth](https://github.com/cylin-gn/Tel2Veh/blob/main/Model/save/exp202401251640_0.pth)
- Trained for Fusion & Third-STGNN (STGNN-3): [exp202401251640_0.pth](https://github.com/cylin-gn/Tel2Veh/blob/main/Model/save/exp202401251640_0.pth)


## Model Training

- Training Extractors for GCT flow prediction and Vehicle flow prediction, similar to traffic speed prediction.
  For using Graph Wavenet as an example, please follow the instruction in [Graph Wavenet](https://github.com/nnzhan/Graph-WaveNet) 
- Put the pre-trained-well extractors model in:
```
./Model/save
```
- Please set the location of the dataset and graph structure file in `argparse.ArgumentParser()` of `parameter.py`

For GCT Flow extractors, please set in the:
```
### GCT ###
...
```

For Vehicle Flow extractors, please set in the:
```
### CCTV ###
...
```

For Stage Two, please set in the:
```
### Fusion ###
...
```

And put all codes together to run the training process.

Or directly run the `Jupyter Notebook`:

```
Framework(with GENET).ipynb
```

for Stage Two training with our provided pre-trained extractors.

### Example of Integrating STGNN into the Framework for Prediction: We exclude this vehicle flow during training to simulate a camera-free scenario.
![dataset](https://github.com/cylin-gn/Tel2Veh/blob/main/Figure/output.png)


