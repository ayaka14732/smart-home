from config import *
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_application_credentials

import logging
import io
import struct
import subprocess
from tempfile import NamedTemporaryFile

from google.cloud import speech_v1p1beta1, texttospeech
import numpy as np
from openai import OpenAI
import pvporcupine
from pvrecorder import PvRecorder
import pydub.utils
import requests
import wave

silence_threshold = 0.015  # depends on environment noise
required_silence_seconds = 3.
initial_prompt = '你係一個香港人，係講廣東話嘅助理。唔好講太多嘢，所有問題請非常簡要回答。但如果對方心情唔好，請你認真安慰。'

class GPTClient:
    def __init__(self) -> None:
        self.messages = [{'role': 'system', 'content': initial_prompt}]

    def reply(self, msg: str) -> str:
        self.messages.append({'role': 'user', 'content': msg})
        completion = openai_client.chat.completions.create(model='gpt-3.5-turbo', messages=self.messages)  # type: ignore
        model_response: str = completion.choices[0].message.content  # type: ignore
        logger.info(f'Got response from ChatGPT: {model_response}')
        self.messages.append({'role': 'system', 'content': model_response})
        return model_response

    def reset(self) -> None:
        self.messages = [{'role': 'system', 'content': initial_prompt}]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s [%(funcName)s] %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

google_tts_client = texttospeech.TextToSpeechClient()
google_tts_voice = texttospeech.VoiceSelectionParams(language_code='zh-HK', ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
google_tts_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, speaking_rate=1.1)

google_speech_client = speech_v1p1beta1.SpeechClient()
google_speech_config = speech_v1p1beta1.RecognitionConfig(encoding=speech_v1p1beta1.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code='yue-Hant-HK')

openai_client = OpenAI(api_key=openai_api_key)

keyword = 'hey siri'
keyword_paths = [pvporcupine.KEYWORD_PATHS[keyword]]
porcupine = pvporcupine.create(access_key=porcupine_access_key, keyword_paths=keyword_paths)

recorder = PvRecorder(frame_length=porcupine.frame_length)
recorder.start()

gpt_client = GPTClient()

def block_until_woken_up() -> None:
    recorder.start()
    while True:
        pcm = recorder.read()
        res = porcupine.process(pcm)
        if res >= 0:
            logger.info(f'Detected {keyword}')
            recorder.stop()
            return

def play_audio(audio_data: bytes) -> None:
    player = pydub.utils.get_player_name()
    with NamedTemporaryFile('w+b', suffix='.wav') as f:
        f.write(audio_data)
        with open(os.devnull, 'w') as fp:
            subprocess.call([player, '-nodisp', '-autoexit', '-hide_banner', f.name], stdout=fp, stderr=fp)

def text_to_speech(text: str) -> None:
    synthesis_input = texttospeech.SynthesisInput(text=text)
    response = google_tts_client.synthesize_speech(input=synthesis_input, voice=google_tts_voice, audio_config=google_tts_config)
    audio_data = response.audio_content
    play_audio(audio_data)

def transcribe(audio_data) -> str:
    audio = speech_v1p1beta1.RecognitionAudio(content=audio_data)
    request = speech_v1p1beta1.RecognizeRequest(config=google_speech_config, audio=audio)
    response = google_speech_client.recognize(request=request)
    if len(response.results) == 0:
        logger.info('No speech detected')
        return ''
    text = response.results[0].alternatives[0].transcript
    logger.info(f'Detected speech: {text}')
    return text

def light_off() -> None:
    logger.info('Calling the API for switching the light off')
    headers = {'Content-Type': 'application/json'}
    data = {'id': 0, 'on': False}
    response = requests.post(light_switch_endpoint, headers=headers, json=data)
    logger.info(f'Got API response {response.json()}')

def light_on() -> None:
    logger.info('Calling the API for switching the light on')
    headers = {'Content-Type': 'application/json'}
    data = {'id': 0, 'on': True}
    response = requests.post(light_switch_endpoint, headers=headers, json=data)
    logger.info(f'Got API response {response.json()}')

def tv_off() -> None:
    logger.info('Calling the API for switching the TV off')
    headers = {'Content-Type': 'application/json'}
    data = {'id': 0, 'on': False}
    response = requests.post(tv_switch_endpoint, headers=headers, json=data)
    logger.info(f'Got API response {response.json()}')

def tv_on() -> None:
    logger.info('Calling the API for switching the TV on')
    headers = {'Content-Type': 'application/json'}
    data = {'id': 0, 'on': True}
    response = requests.post(tv_switch_endpoint, headers=headers, json=data)
    logger.info(f'Got API response {response.json()}')

def pcm_to_wav_data(pcm) -> bytes:
    frames = struct.pack('h' * len(pcm), *pcm)

    buffer = io.BytesIO()
    wavfile = wave.open(buffer, 'w')
    wavfile.setparams((1, 2, recorder.sample_rate, recorder.frame_length, 'NONE', 'NONE'))
    wavfile.writeframes(frames)

    data = buffer.getvalue()
    return data

def pcm_to_array(pcm) -> np.ndarray:
    data = pcm_to_wav_data(pcm)
    arr = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
    return arr

def is_silence(arr: np.ndarray) -> bool:
    return np.mean(np.abs(arr)) < silence_threshold

def record_until_silence() -> bytes:
    recorder.start()
    silence_duration = 0.0
    pcms = []
    while True:
        pcm = recorder.read()
        pcms += pcm
        arr = pcm_to_array(pcm)
        if is_silence(arr):
            silence_duration += len(pcm) / recorder.sample_rate
            if silence_duration >= required_silence_seconds:
                recorder.stop()
                logger.info(f'{required_silence_seconds} seconds of silence detected, stopping recording')
                wav_data = pcm_to_wav_data(pcms)
                return wav_data
        else:
            silence_duration = 0.0

def main_loop() -> None:
    logger.info('Process initialised, listening...')

    while True:
        block_until_woken_up()
        text_to_speech('咩事？')
        gpt_client.reset()

        while True:
            wav_data = record_until_silence()
            text = transcribe(wav_data)

            if not text.strip():
                text_to_speech('係噉先喇，加油！')
                break

            if '退出' in text or '關閉你自己' in text:
                return

            if '開燈' in text:
                light_on()
                text_to_speech('已經幫你開咗燈喇')
                break

            if '熄燈' in text or '關燈' in text or '熄咗盞燈佢' in text:
                light_off()
                text_to_speech('已經幫你熄咗燈喇')
                break

            if '開電視' in text:
                tv_on()
                text_to_speech('已經幫你開咗電視喇')
                break

            if '關電視' in text:
                tv_off()
                text_to_speech('已經幫你關咗電視喇')
                break

            model_response = gpt_client.reply(text)
            text_to_speech(model_response)

def main():
    try:
        main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.delete()
        porcupine.delete()

if __name__ == '__main__':
    main()
