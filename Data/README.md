# Dataset Providing

## Raw Dataset

The raw data is now provided at : 
```
./Raw
```

- The original CSV file for GCT flow is available at: [GCT flow.csv](./Data/Raw/GCT_Flow.csv)

Here is an example:

|        Date         | Road Segment 1 | ...  | Road Segment 49 | 
|:-------------------:|:--------------:|:--------------:|:--------------|
|         ...         |    ...         |    ...         |   
| 08-28 18:55 |  81        |  ...        |   228        |   
| 08-28 19:00 |  50        |  ...        |   186        |    
| 08-29 06:00 |  20        |  ...         |   31        |  
|         ...         |    ...         |    ...         |   

- The original CSV file for Vehicle flow is available at: [Vehicle flow.csv](./Data/Raw/Vehicle_Flow_Raw.csv)


|        Date         |  Cam1 | ... | Cam6 | 
|:-------------------:|:--------------:|:--------------:|:--------------:|
|         ...         |    ...         |    ...         |    ...         |    ...        |    ...        |    ...        |
| 08-28 18:55 |     151         |    ...        |   158        |
| 08-28 19:00 |        138         |     ...        |   177        |
| 08-29 06:00 |     38         |    ...        |   14        |
|         ...            |      ...        |   ...        |

- To generate the **train/val/test datasets** for each type of GCT flow as {train,val,test}.npz, please follow the [script](https://github.com/liyaguang/DCRNN/blob/master/scripts/generate_training_data.py),
using the CSV files provided above.

## train/test/valid dataset

The train/test/val data is now provided at : 
- For GCT flow:
```
./hsin_49_GCT_0600_1900_rename
```
- For Vehicle flow:
```
./hsin_9_CCTV_0600_1900_rename
```

## Road Network's Graph Structure
As the implementation is based on pre-calculated distances between road sections, we provided the CSV file with road section distances and IDs at: 
- GCT Flow: [Road Section Distance](./Data/Raw/GCT_Roads_Distance.txt). 
- Vehicle Flow: [Road Section Distance](./Data/Raw/Vehicle_Flow_Roads_Distance.txt). 

Run the [script](https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py) to generate the Graph Structure based on the "Road Section Distance" file provided above.

The `processed Graph Structure of Road Section Network` is available at: 
- GCT Flow: [road network structure file(.pkl)](https://github.com/cylin-gn/Tel2Veh/blob/main/Data/hsin_49_GCT_0600_1900_rename/adj_mat_49_corrected.pkl)
- Vehicle Flow: [road network structure file(.pkl)](https://github.com/cylin-gn/Tel2Veh/blob/main/Data/hsin_9_CCTV_0600_1900_rename/adj_mat_9.pkl)
