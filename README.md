# ConvNetFaceDetection
Creation of an app to detect face in images.  

### Setup and/or start a python virtual environemnt
Since python packages exist in numerous versions, and to avoid dealing with different versions installed conflicts, we're going to use a python
[virtual environement](https://docs.python.org/3/tutorial/venv.html)  
#### Installation
Use the command below to install `python3-venv` :  
```bash
    $ sudo apt-get install python3-venv
```  
#### Create and/or activate a virtual environment
Create a virtual environment named `virtualenv` (ignored by the git, if you choose another name, please add it to the `.gitignore` file !)   
```bash
    $ python3 -m venv virtualenv
```  
**NB :** You only need to create your virtual environment once !  
Activate the virtual environment **each time you want to run the project** :
```bash
    $ source virtualenv/bin/activate
```  
To install the required python packages needed to run the project, or new added ones use the command below once your virtual environment is activated :  
```bash
    $ pip install -r requirements.txt
```  
#### Install new packages
To install a new package use `pip install pckg_name` (once your virtual environement is activated!)  
Don't forget to add the new package to `requirements.txt` using the command in the **project root** :    
```bash
    $ pip freeze > requirements.txt
```  
### Requirements
* Installing python3 and python3-venv
* Creating and activating a virtual environement
* Installing the required packages contained in `requirements.txt` in the virtual environement 


e