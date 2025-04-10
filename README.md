# SC4062 GEN AI Visual Content Generation Group Project

## **Repository Directories**
- ```data```: Folder containing the 
- ```train_lora.py:``` Python File used for fine-tuning Stable Diffusion Model
- ```eval.py:``` Python File used to evaluate the Fine-tuned Stable Diffusion Model
- ```requirements.txt:``` Containing all the dependencies required to run the code

```Note: Ensure that you change the directories inside the python files to be able to run the code```

---

## **Local Environment Setup Instructions**

### **Prerequisites**
Ensure you have the following installed on your system:
- Python 3.10 or higher
- Conda or Miniconda for virtual environment management

### **Instructions to install Miniconda with Python on your system**

### For Linux (Preferred)
```bash
# Download Miniconda Installer For Linux (Ubuntu/Debian, Fedora, Arch)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the installer
bash ~/Miniconda3-latest-Linux-x86_64.sh

# Close and reopen your terminal, then run the following code to activate Miniconda
source ~/.bashrc

# Verify Installation of Miniconda
conda --version
```

For Windows and macOS installation, kindly refer to the following [Official Anconda Website](https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation) for more information on the installation

---


### **1. Create a Virtual Environment**
Use Conda or Miniconda to create and activate a virtual environment (downgrade or upgrade Python Version when necessary):

### **Using Conda (Preferred)**
```bash
# Creating virtual environment
conda create --name gen_ai python=3.10

# Activating environment
conda activate gen_ai 
```

### **Using Python's venv**
```bash
# Creating virtual environment
python -m venv gen_ai

"""
Steps on Virtual Environment management
""" 
gen_ai\Scripts\activate # Activate env on Windows
source gen_ai/bin/activate # Activate env on macOS / Linus

# Deactivate environment
deactivate
```
---

### **2. Install Required Libraries**
Install all required libraries using the provided requirements.txt file:
```bash
pip install -r requirements.txt
```
