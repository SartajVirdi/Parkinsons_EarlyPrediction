import subprocess
import threading
import os

def run_csv_script():
    subprocess.run(['python3', 'get_csv.py'], check=True)

def run_wav_script():
    subprocess.run(['python3', 'get_wav.py'], check=True)

def run_final_process():
    subprocess.run(['python3', 'final_process_2.py'], check=True)

csv_thread = threading.Thread(target=run_csv_script)
wav_thread = threading.Thread(target=run_wav_script)

csv_thread.start()
wav_thread.start()

csv_thread.join()
wav_thread.join()

run_final_process()

print("\nPipeline completed successfully ✅")
