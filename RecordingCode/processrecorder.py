
import subprocess 
import os 
folder_path = "yolov5models"
import time
import resource
import socket

receiver_ip = '192.168.0.143'
receiver_port = 5000

folder_path = "yolov8diorship"

for filename in os.listdir(folder_path):
	if os.path.isfile(os.path.join(folder_path, filename)):
		print(filename)


def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)
'''
cmd =f'python detectyolov5timeall.py'
print(cmd)
subprocess.run(cmd, shell=True)

cmd =f'python detectyolov5timeship.py'
print(cmd)
subprocess.run(cmd, shell=True)

cmd =f'python detectyolov8timeall.py'
print(cmd)
subprocess.run(cmd, shell=True)

cmd =f'python nonmsdetectyolov8timeship.py'
print(cmd)
subprocess.run(cmd, shell=True)	
'''


def record_time(command):
    start_wall_time = time.time()
    start_cpu_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime

    subprocess.run(command, shell=True)

    end_cpu_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    end_wall_time = time.time()

    elapsed_wall_time = end_wall_time - start_wall_time
    elapsed_cpu_time = end_cpu_time - start_cpu_time

    return elapsed_wall_time, elapsed_cpu_time




# === Model Groups === you could run an os command to get the filepaths however i found that when testing on many diffrent OS its best to explicitly give the full paths as these commands can changed based on the device
YOLOv5ALL = ["yolov5n-diorall-fp16", "yolov5s-diorall-int8", "yolov5s-diorall-fp16", "yolov5n-diorall-int8"]
YOLOv5SHIP = ["yolov5s-diorship-int8", "yolov5n-diorship-int8", "yolov5n-diorship-fp16", "yolov5s-diorship-fp16"]
YOLOv8SHIP = ["diorshipyolov8s_float16", "diorshipyolov8s_int8", "diorshipyolov8n_float16", "diorshipyolov8n_int8"]
YOLOv8ALL = ["diorallyolov8s_float16", "diorallyolov8s_int8", "diorallyolov8n_float16", "diorallyolov8n_int8"]
TPUAll = ["yolov5n-diorall-int8-256", "yolov5n-diorall-fp16-256"]
TPUSHIP = ["yolov5n-diorship-int8-448", "yolov5n-diorship-fp16-448"]

 




def run_psrecord(script, model, flag, size):
    for i in range(3):
        cmd = (
            f'psrecord "python {script} yolov5models/{model} {flag} {size}" '
            f'--log "processrecordings/{model}{i}.txt" --include-children'
        )
        print(cmd)
        subprocess.run(cmd, shell=True)


def run_powerinfer(script, path_prefix, model, size=None):
    """Run power inference script 3 times with an optional size argument."""
    for i in range(3):
        if size is not None:
            cmd = f'python {script} {path_prefix}/{model} {size}'
        else:
            cmd = f'python {script} {path_prefix}/{model}'
        print(cmd)
        subprocess.run(cmd, shell=True)

# Run YOLOv5ALL and YOLOv5SHIP (with size=0)

for model in TPUAll:
    run_powerinfer('detectyolov5powerinferall.py', 'yolov5models', model, 0)

for model in TPUSHIP:
    run_powerinfer('detectyolov5powerinfership.py', 'yolov5models', model, 0)

for model in YOLOv5ALL:
    run_powerinfer('detectyolov5powerinferall.py', 'yolov5models', model, 0)

for model in YOLOv5SHIP:
    run_powerinfer('detectyolov5powerinfership.py', 'yolov5models', model, 0)

# Run YOLOv8ALL and YOLOv8SHIP (no size argument)
for model in YOLOv8ALL:
    run_powerinfer('detectyolov8powerinferall.py', 'yolov8diorall', model)

for model in YOLOv8SHIP:
    run_powerinfer('detectyolov8powerinfership.py', 'yolov8diorship', model)


'''
for i in range(3):
	cmd =f'psrecord "python detectyolov5all.py yolov5models/{TPUAll[0]} True 256" --log "processrecordings/"{TPUAll[0]}{i}.txt --include-children' 
	print(cmd)
	subprocess.run(cmd, shell=True)	

for i in range(3):
	cmd =f'psrecord "python detectyolov5all.py yolov5models/{TPUAll[1]} True 256" --log "processrecordings/"{TPUAll[1]}{i}.txt --include-children' 
	print(cmd)
	subprocess.run(cmd, shell=True)

for i in range(3):
	cmd =f'psrecord "python detectyolov5ship.py yolov5models/{TPUSHIP[0]} True 448" --log "processrecordings/"{TPUSHIP[0]}{i}.txt --include-children' 
	print(cmd)
	subprocess.run(cmd, shell=True)	

for i in range(3):
	cmd =f'psrecord "python detectyolov5ship.py yolov5models/{TPUSHIP[1]} True 448" --log "processrecordings/"{TPUSHIP[1]}{i}.txt --include-children' 
	print(cmd)
	subprocess.run(cmd, shell=True)

for model in YOLOv5ALL:
	for i in range(3):
		cmd =f'psrecord "python detectyolov5all.py yolov5models/{model} 0" --log "processrecordings/"{model}{i}.txt --include-children' 
		print(cmd)
		subprocess.run(cmd, shell=True)
		
for model in YOLOv5SHIP:
	for i in range(3):
		cmd =f'psrecord "python detectyolov5ship.py yolov5models/{model} 0" --log "processrecordings/"{model}{i}.txt --include-children' 
		print(cmd)
		subprocess.run(cmd, shell=True)



for model in YOLOv8SHIP:
	for i in range(3):
		cmd =f'psrecord "python detectyolov8ship.py yolov8diorship/{model}" --log "processrecordings/"{model}{i}.txt --include-children' 
		print(cmd)
		subprocess.run(cmd, shell=True)


for model in YOLOv8ALL:
	for i in range(3):
		cmd =f'psrecord "python detectyolov8all.py yolov8diorall/{model}" --log "processrecordings/"{model}{i}.txt --include-children' 
		print(cmd)
		subprocess.run(cmd, shell=True)		
'''

def run_psrecord(models, script, path_prefix, extra_arg=None):
    """
    Run psrecord on one or more models.

    Args:
        models (str or list): model name(s)
        script (str): script to execute
        path_prefix (str): directory containing the models
        extra_arg (str or None): optional argument(s) to pass to the script
    """
    if isinstance(models, str):
        models = [models]

    for model in models:
        for i in range(3):
            cmd = f'psrecord "python {script} {path_prefix}/{model}'
            if extra_arg:
                cmd += f' {extra_arg}'
            cmd += f'" --log processrecordings/{model}{i}.txt --include-children'
            print(cmd)
            subprocess.run(cmd, shell=True)

# TPU models
run_psrecord(TPUAll, 'detectyolov5all.py', 'yolov5models', 'True 256')
run_psrecord(TPUSHIP, 'detectyolov5ship.py', 'yolov5models', 'True 448')

# YOLOv5
run_psrecord(YOLOv5ALL, 'detectyolov5all.py', 'yolov5models', '0')
run_psrecord(YOLOv5SHIP, 'detectyolov5ship.py', 'yolov5models', '0')

# YOLOv8 (no extra arg)
run_psrecord(YOLOv8SHIP, 'detectyolov8ship.py', 'yolov8diorship')
run_psrecord(YOLOv8ALL, 'detectyolov8all.py', 'yolov8diorall')


'''
#####################################################################################################################
for model in YOLOv5SHIP:
	for i in range(3):
		cmd =f'python detectyolov5powerinfership.py yolov5models/{model} 0'
		print(cmd)
		subprocess.run(cmd, shell=True)
		
for model in YOLOv5ALL:
	for i in range(3):
		cmd =f'python detectyolov5powerinferall.py yolov5models/{model} 0'
		print(cmd)
		subprocess.run(cmd, shell=True)

for model in YOLOv8SHIP:
	for i in range(3):
		cmd =f'python detectyolov8powerinfership.py yolov8diorship/{model}' 
		print(cmd)
		subprocess.run(cmd, shell=True)



for model in YOLOv8ALL:
	for i in range(3):
		cmd =f'python detectyolov8powerinferall.py yolov8diorall/{model}' 
		print(cmd)
		subprocess.run(cmd, shell=True)		

for i in range(3):
	cmd =f'python detectyolov5powerinferall.py yolov5models/{TPUAll[0]}  256' 
	print(cmd)
	subprocess.run(cmd, shell=True)
for i in range(3):
	cmd =f'python detectyolov5powerinferall.py yolov5models/{TPUAll[1]}  256' 
	print(cmd)
	subprocess.run(cmd, shell=True)
	
for i in range(3):
	cmd =f'python detectyolov5powerinfership.py yolov5models/{TPUSHIP[0]}  448' 
	print(cmd)
	subprocess.run(cmd, shell=True)
	
for i in range(3):
	cmd =f'python detectyolov5powerinfership.py yolov5models/{TPUSHIP[1]}  448' 
	print(cmd)
	subprocess.run(cmd, shell=True)



'''		


 


'''

for model in YOLOv5SHIP:
	for i in range(3):
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((receiver_ip, receiver_port))
		message = 'starta'+model
		print(message)
		sock.sendall(message.encode())
		sock.close()
		cmd =f'python detectyolov5ship.py yolov5models/{model} 0'
		subprocess.run(cmd, shell=True)
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((receiver_ip, receiver_port))
		message = 'stop'
		msg = message.encode()
		sock.sendall(message.encode())
		sock.close()
		
for model in YOLOv5ALL:
	for i in range(3):
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((receiver_ip, receiver_port))
		message = 'starta'+model
		print(message)
		sock.sendall(message.encode())
		sock.close()
		cmd =f'python detectyolov5all.py yolov5models/{model} 0'
		subprocess.run(cmd, shell=True)
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((receiver_ip, receiver_port))
		message = 'stop'
		msg = message.encode()
		sock.sendall(message.encode())
		sock.close()


for model in YOLOv8SHIP:
	for i in range(3):
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((receiver_ip, receiver_port))
		message = 'starta'+model
		print(message)
		sock.sendall(message.encode())
		sock.close()
		cmd =f'python detectyolov8ship.py yolov8diorship/{model}'
		subprocess.run(cmd, shell=True)
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((receiver_ip, receiver_port))
		message = 'stop'
		msg = message.encode()
		sock.sendall(message.encode())
		sock.close()


for model in YOLOv8ALL:
	for i in range(3):
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((receiver_ip, receiver_port))
		message = 'starta'+model
		print(message)
		sock.sendall(message.encode())
		sock.close()
		cmd =f'python detectyolov8all.py yolov8diorall/{model}'
		subprocess.run(cmd, shell=True)
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((receiver_ip, receiver_port))
		message = 'stop'
		msg = message.encode()
		sock.sendall(message.encode())
		sock.close()
        
         
for model in TPUAll:
	for i in range(3):
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((receiver_ip, receiver_port))
		message = 'starta'+model
		print(message)
		sock.sendall(message.encode())
		sock.close()
		cmd =f'python detectyolov5all.py yolov5models/{model} 256'
		subprocess.run(cmd, shell=True)
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((receiver_ip, receiver_port))
		message = 'stop'
		msg = message.encode()
		sock.sendall(message.encode())
		sock.close()
 
for model in TPUSHIP:
	for i in range(3):
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((receiver_ip, receiver_port))
		message = 'starta'+model
		print(message)
		sock.sendall(message.encode())
		sock.close()
		cmd =f'python detectyolov5ship.py yolov5models/{model} 448'
		subprocess.run(cmd, shell=True)
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((receiver_ip, receiver_port))
		message = 'stop'
		msg = message.encode()
		sock.sendall(message.encode())
		sock.close()



        
'''

def send_message(message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((receiver_ip, receiver_port))
    sock.sendall(message.encode())
    sock.close()


#this needs work extra arg super dumb implementation and get it to wrk with edge tpu models
def run_models(models, script, path_prefix, extra_arg=True):
    for model in models:
        for _ in range(3):
            send_message('starta' + model)
            print('starta' + model)
            cmd = f'python {script} {path_prefix}/{model}'
            if extra_arg:
                cmd += ' 0'
            subprocess.run(cmd, shell=True)
            send_message('stop')

run_models(YOLOv5SHIP, 'detectyolov5ship.py', 'yolov5models')
run_models(YOLOv5ALL, 'detectyolov5all.py', 'yolov5models')
run_models(YOLOv8SHIP, 'detectyolov8ship.py', 'yolov8diorship', extra_arg=False)
run_models(YOLOv8ALL, 'detectyolov8all.py', 'yolov8diorall', extra_arg=False)







 
'''
for i in range(3):
	cmd =f'python detectyolov5all.py yolov5models/{TPUAll[0]} 256'
	elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
	with open(f'logfiletime/total-time{TPUAll[0]}.txt','a') as f:
		f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")

for i in range(3):
	cmd =f'python detectyolov5all.py yolov5models/{TPUAll[1]} 256'
	elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
	with open(f'logfiletime/total-time{TPUAll[1]}.txt','a') as f:
		f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")

for i in range(3):
	cmd =f'python detectyolov5ship.py yolov5models/{TPUSHIP[0]} 448'
	elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
	with open(f'logfiletime/total-time{TPUSHIP[0]}.txt','a') as f:
		f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")

for i in range(3):
	cmd =f'python detectyolov5ship.py yolov5models/{TPUSHIP[1]} 448'
	elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
	with open(f'logfiletime/total-time{TPUSHIP[1]}.txt','a') as f:
		f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")
'''


'''

for model in YOLOv5ALL:
	for i in range(3):
		cmd =f'python detectyolov5all.py yolov5models/{model} 0'
		elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
		with open(f'logfiletime/total-time{model}.txt','a') as f:
			f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")
for model in YOLOv5SHIP:
	for i in range(3):
		cmd =f'python detectyolov5ship.py yolov5models/{model} 0'
		elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
		with open(f'logfiletime/total-time{model}.txt','a') as f:
			f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")


for model in YOLOv8SHIP:
	for i in range(3):
		cmd =f'python detectyolov8ship.py yolov8diorship/{model} True'
		elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
		with open(f'logfiletime/total-time{model}.txt','a') as f:
			f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")

for model in YOLOv8ALL:
	for i in range(3):
		cmd =f'python detectyolov8all.py yolov8diorall/{model} True'
		elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
		with open(f'logfiletime/total-time{model}.txt','a') as f:
			f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")
'''

def run_and_log(models, script, path_prefix, extra_arg=None):    
    for model in models:
        for _ in range(3):
            cmd = f'python {script} {path_prefix}/{model}'
            if extra_arg is not None:
                cmd += f' {extra_arg}'
            elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
            with open(f'logfiletime/total-time{model}.txt', 'a') as f:
                f.write(f"total {elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")


run_and_log(TPUAll, 'detectyolov5all.py','yolov5models', '256')
 
run_and_log(TPUSHIP, 'detectyolov5ship.py','yolov5models', '448')

 
run_and_log(YOLOv5ALL, 'detectyolov5all.py', 'yolov5models', '0')
run_and_log(YOLOv5SHIP, 'detectyolov5ship.py', 'yolov5models', '0')
run_and_log(YOLOv8SHIP, 'detectyolov8ship.py', 'yolov8diorship', 'True')
run_and_log(YOLOv8ALL, 'detectyolov8all.py', 'yolov8diorall', 'True')





################################

################################################






