## What is this project about?
In this project, an interactive web application is deployed that allow the user to select a dataset and a classifier.
The parameters of the classifiers can be changed freely.
You can run the app on your localhost or use the deployed webaoo.

## Demo:
![webapp_demo](https://user-images.githubusercontent.com/32487291/142691293-3c2e3bb5-2bb7-4245-993c-1de268000a18.gif)

## Try it out yourself:
Launch App in Browser: [Start App](https://share.streamlit.io/anelmusic/interactivemachinelearningwebapp/main/app/main.py)


## Directory structure
```bash
app/
├── appmanager.py             - manager interaction between ui and classification
└── classifier_config.py      - classifier enum definition
├── dataset_config.py         - dataset enum definition
├── main.py                   - entry point
├── ui.py                     - user interface
├── config.py                 - configuration setup
├── wrapped_cli_tool.py       - streamlit cli
```

## Basic Workflows

#### 1. Fork project.
(https://github.com/AnelMusic/interactiveMachineLearningWebApp/fork)

#### 2. Set up environment. (Routine defined in Makefile)
> This will automatically install all dependencies defined in requirements.txt
```bash
make venv
source venv/bin/activate
```
#### 3. Download to data (Routine defined in CLI (see app/cli.py))
```bash
dvc pull
```
#### 4. Start Webapp on your localhost
> This will automatically load the best model (with respect to accuracy) from all MLflow experint runs
```bash
mlapp streamlit
```

<br/>
<br/>


## Helpful Makefile Commands
##### Remove not needed files:
```bash
make clean
```
##### Run Code-Formatting:
```bash
make style
```
<br/>
<br/>


## Fell free to connect:
Anel Music– [@LinkedIn](https://www.linkedin.com/in/anelmusic/) – anel.music@web.de

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/AnelMusic/](https://github.com/AnelMusic/)
