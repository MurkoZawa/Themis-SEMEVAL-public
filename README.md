# Themis: Image-Text classification by leveraging LLM


### Architecture Overview
<div align="center">
  <img src="https://github.com/demon-prin/Themis-SEMEVAL-public/blob/main/content/themispm.png" width="35%" height="35%"/>
</div><br/>

### Features
* Simple to train architecture for multimodal text and image classification
* Top 5 methods for  [[`SEMEVAL 2024 SHARED TASK ON "MULTILINGUAL DETECTION OF PERSUASION TECHNIQUES IN MEMES"`](https://propaganda.math.unipd.it/semeval2024task4/index.html)]
* Fully customizable
* Simple to integrate into already existing code
* Compatible with Huggingface LLM and ImageEncoders
  

## Setup
> [!TIP]
> Before using the code, make sure to follow these setup instructions:

### [STEP 1] Requirements install

```bash
pip install -r requirements.txt
```

### [STEP 2]  Select Model Configuration

The full list of the currently supported model configurations can be found in the train_all.bat file while train_search.bat highlights the full set of possible hyperparameters

### [STEP 3]  Edit the train.py to match your data

A small usage example suitable for the SEMEVAL challenge data is provided in the "train.py" file feel free to modify it according to your needs 

### [STEP 4]  Train a Themis model

After selecting hyperparameters, model configuration, and train file configuration you can find the trained models under the "outputs" folder

### [STEP 5]  Test a Themis model

After training your model you can evaluate your model by using the "eval.py" file, be sure to follow the same dataset structure of your "train.py" file, small usage examples are provided in the "eval_all.bat" file

## License

This project is licensed under the [MIT License](LICENSE).



## <a name="CitingThemis"></a>Citing Themis

If you use Themis in your research and want to cite our work a paper is on the way!

