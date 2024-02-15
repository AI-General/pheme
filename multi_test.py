import asyncio
import concurrent.futures
import io
import wave
import uvicorn
import numpy as np
from datetime import datetime
from fastapi import Request, FastAPI
from fastapi.responses import StreamingResponse
from transformer_infer import PhemeClient
import time
import os

# Set an environment variable
os.environ['HUGGING_FACE_HUB_TOKEN'] = '<YOUR_TOKEN>'

class PhemeArgs:
    text = 'I gotta say, I would never expect that to happen!'
    manifest_path = 'demo/manifest.json'
    outputdir = 'demo'
    featuredir = 'demo/'
    text_tokens_file = 'ckpt/unique_text_tokens.k2symbols'
    t2s_path = 'ckpt/t2s/'
    s2a_path = 'ckpt/s2a/s2a.ckpt'
    target_sample_rate = 16000
    temperature = 0.7
    top_k = 210
    voice = 'male_voice'


model0 = PhemeClient(PhemeArgs())
model1 = PhemeClient(PhemeArgs())

text0 = "I see you received a callback from us. Our outreach department was reaching out regarding our hardship program."
text1 = "I see you received a callback from us. Our outreach department was reaching out regarding our hardship program. "
# model.infer(text)

# async def ainfer(text, index=0, executor=None):
#     print(index, "start")
#     loop = asyncio.get_running_loop()
#     await loop.run_in_executor(executor, model.infer, text)
#     print(index, "end")


# async def main():
#     # Create a ProcessPoolExecutor
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         await asyncio.gather(
#             ainfer(text0, index=0, executor=executor),
#             ainfer(text1, index=1, executor=executor)
#         )
# # This is how you should call your main coroutine


# # Serial
# # start_time = time.time()
# # model.infer(text0)
# # model.infer(text1)
# # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# # print("Serial", time.time() - start_time)
# # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# start_time = time.time()
# asyncio.run(main())
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print("Parallel", time.time() - start_time)
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

import multiprocessing

def process0(text):
    print(0, "start")
    model0.infer(text)
    print(0, "end")

def process1(text):
    print(1, "start")
    model1.infer(text)
    print(1, "end")
    
if __name__ == '__main__':
    # Serial
    start_time = time.time()
    model0.infer(text0)
    model1.infer(text1)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Serial", time.time() - start_time)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    multiprocessing.set_start_method('spawn')
    p1 = multiprocessing.Process(target=process0, args=(text0,))
    p2 = multiprocessing.Process(target=process1, args=(text1,))

    start_time = time.time()
    p1.start()
    p2.start()

    p1.join()
    p2.join()
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Parallel", time.time() - start_time)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
