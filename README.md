# Neural Machine Translation

Multi-model implementation for neural machine translation from Nepali to English.

## Requirements

Install required fundamental packages:

```
pip3 install pandas
pip3 install torch
pip3 install tqdm
pip3 install loguru
```

## Prepare your dataset

Prepare your dataset, as intended, before proceeding to the task.

If you wish to go with the same dataset as here, follow the guidelines here.

* Download dataset from [here](https://drive.google.com/drive/folders/1hV2W2xXTaBsW5QEMlQlqkWfjOsZCsDTG?usp=sharing)
* Prepare your dataset with the following command:
  `python3 data.py`
* You may delete the previous `.txt` files (be careful what you delete)

Train your model with:

`python3 train.py`


## Configuration

You can always manage your project configuration using the file `config.py`.

`Remember:` This should be done before training your model or it may not work as expected.