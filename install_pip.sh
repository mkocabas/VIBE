#!/usr/bin/env bash

echo "Creating virtual environment"
python3.7 -m venv vibe-env
echo "Activating virtual environment"

source $PWD/vibe-env/bin/activate

$PWD/vibe-env/bin/pip install numpy torch torchvision
$PWD/vibe-env/bin/pip install -r requirements.txt
