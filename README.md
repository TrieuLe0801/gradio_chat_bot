# gradio_chat_bot:
A demo chatbot with Gradio and LLM. Thanks to Sentdex: https://www.youtube.com/@sentdex
# Prequirements:
- Python>=3.7
- OS: MacOS, Linux
- Set up your environment (MacOS, Linux)
  - Virtualenv:
    ```
    python3 -m pip install --user virtualenv
    python3 -m venv env
    source env/bin/activate # Activate your environment
    which python # Check your python environment
    python3 -m pip install -r requirements.txt
    deactivate # Only used to deactivate the environment
    ```
  - Conda:
    - Install Miniconda [here](https://docs.conda.io/projects/miniconda/en/latest/)
      ```
      conda create --name venv python=<YOUR_PYTHON_VERSION> -y
      conda activate venv
      pip install -r requirements.txt
      conda deactivate
      ```
  # Start
  ```
  python gradio_chat_interface.py
  ```
    
