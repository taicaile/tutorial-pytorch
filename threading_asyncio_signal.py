import signal
import time
import threading
import subprocess
import datetime
import asyncio

proc = None
def key_interrupt(signum, frame):
    global key_in
    print(signum, frame)
    print("key interrupt detected")
    if proc:
        proc.terminate()
    else:
        raise KeyboardInterrupt
# signal.signal(signal.SIGTERM, key_interrupt)
signal.signal(signal.SIGINT, key_interrupt)

def thread_worker():
    global proc
    try:
        proc = subprocess.Popen('tail -f requirements.txt', 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE,
                                shell=True)
        while True:
            line = proc.stdout.readline()
            print(line.decode().strip())
            if not line:
                break
        proc.wait()
        proc = None
    except Exception as e:
        print(e)

async def worker():
    while True:
        print("from worker")
        time.sleep(1)
loop = asyncio.get_event_loop()
p = threading.Thread(target=thread_worker)
p.start()
loop.create_task(worker())
loop.run_forever()
print("wait for thread finished.")
p.join()
print("exit.")

while True:
    print("wait for terminate")
    time.sleep(1)