#!/usr/bin/env bash
#!/usr/bin/env zsh

# Create virtual enviroment to prevent clashing dependencies
python3 -m venv venv

# Create a virtual enviroment
source venv/bin/activate

# Install all required dependencies
pip3 install -r requirements.txt

# Be sure to enable permisssions using the command: chmod u+x setup.sh
