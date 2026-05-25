import json
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Generative Retrieval for Wine Recommendation\n",
                "Tiến hành huấn luyện mô hình siêu tốc bằng Unsloth trên Google Colab. Yêu cầu: Upload toàn bộ folder CD3 lên MyDrive gốc trước khi chạy."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n",
                "!pip install --no-deps xformers \"trl<0.9.0\" peft accelerate bitsandbytes"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "os.chdir('/content/drive/MyDrive/CD3')\n",
                "os.environ['PYTHONIOENCODING'] = 'utf-8'\n",
                "!python src/fine_tune.py --epochs 1 --batch_size 4 --gradient_accumulation_steps 16"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python evaluation/constrained_eval.py --eval_size 12991"
            ]
        }
    ],
    "metadata": {
        "colab": {
            "name": "colab_training.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("colab_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
