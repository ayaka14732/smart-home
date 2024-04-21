# Ayaka’s Smart Home

## Introduction

This repository contains the source code for Ayaka’s smart home AI assistant. The AI assistant is activated by saying "Hey Siri", and communicates with the users in Cantonese. It can turn lights and the TV on and off, as well as chat with the users.

## Run

Create venv:

```sh
python3.12 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

Copy `config.example.py` to `config.py`. Fill in the necessary information.

Then run the script:

```sh
python main.py
```

## Lessons Learned

1. OpenAI Whisper is bad. Use Google Cloud instead.

## TODO

1. Multilingual support
