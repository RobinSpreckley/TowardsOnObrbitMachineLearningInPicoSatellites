
import subprocess 
import os 
folder_path = "yolov5models"
import time
import resource
import socket

receiver_ip = '192.168.0.143'
receiver_port = 5000



for filename in os.listdir(folder_path):
	if os.path.isfile(os.path.join(folder_path, filename)):
		print(filename)
def record_time(command):
    start_wall_time = time.time()
    start_cpu_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime

    subprocess.run(command, shell=True)

    end_cpu_time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    end_wall_time = time.time()

    elapsed_wall_time = end_wall_time - start_wall_time
    elapsed_cpu_time = end_cpu_time - start_cpu_time

    return elapsed_wall_time, elapsed_cpu_time






yolov5allmodels=["yolov5n-diorall-fp16",
"yolov5s-diorall-int8",
"yolov5s-diorall-fp16",
"yolov5n-diorall-int8"]

yolov5shipmodels=["yolov5s-diorship-int8",
"yolov5n-diorship-int8",
"yolov5n-diorship-fp16",
"yolov5s-diorship-fp16"]

yolov8shipmodels=["diorshipyolov8s_float16","diorshipyolov8s_int8",
"diorshipyolov8n_float16",
"diorshipyolov8n_int8"]

diorallmodels=["diorallyolov8s_float16","diorallyolov8s_int8",
"diorallyolov8n_float16",
"diorallyolov8n_int8"]	

folder_path = "yolov8diorship"
edgetpumodels=["yolov5n-diorall-int8-256","yolov5n-diorall-fp16-256"]
edgetpumodels2=["yolov5n-diorship-int8-448","yolov5n-diorship-fp16-448"]

'''
for model in edgetpumodels:
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
'''		
for model in edgetpumodels2:
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

for i in range(3):
	cmd =f'python detectyolov5all.py yolov5models/{edgetpumodels[0]} 256'
	elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
	with open(f'logfiletime/total-time{edgetpumodels[0]}.txt','a') as f:
		f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")

for i in range(3):
	cmd =f'python detectyolov5all.py yolov5models/{edgetpumodels[1]} 256'
	elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
	with open(f'logfiletime/total-time{edgetpumodels[1]}.txt','a') as f:
		f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")

for i in range(3):
	cmd =f'python detectyolov5ship.py yolov5models/{edgetpumodels2[0]} 448'
	elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
	with open(f'logfiletime/total-time{edgetpumodels2[0]}.txt','a') as f:
		f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")

for i in range(3):
	cmd =f'python detectyolov5ship.py yolov5models/{edgetpumodels2[1]} 448'
	elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
	with open(f'logfiletime/total-time{edgetpumodels2[1]}.txt','a') as f:
		f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")
		
		
		

for i in range(3):
	cmd =f'psrecord "python detectyolov5all.py yolov5models/{edgetpumodels[0]} True 256" --log "processrecordings/"{edgetpumodels[0]}{i}.txt --include-children' 
	print(cmd)
	subprocess.run(cmd, shell=True)	

for i in range(3):
	cmd =f'psrecord "python detectyolov5all.py yolov5models/{edgetpumodels[1]} True 256" --log "processrecordings/"{edgetpumodels[1]}{i}.txt --include-children' 
	print(cmd)
	subprocess.run(cmd, shell=True)

for i in range(3):
	cmd =f'psrecord "python detectyolov5ship.py yolov5models/{edgetpumodels2[0]} True 448" --log "processrecordings/"{edgetpumodels2[0]}{i}.txt --include-children' 
	print(cmd)
	subprocess.run(cmd, shell=True)	

for i in range(3):
	cmd =f'psrecord "python detectyolov5ship.py yolov5models/{edgetpumodels2[1]} True 448" --log "processrecordings/"{edgetpumodels2[1]}{i}.txt --include-children' 
	print(cmd)
	subprocess.run(cmd, shell=True)



for i in range(3):
	cmd =f'python detectyolov5powerinferall.py yolov5models/{edgetpumodels[0]}  256' 
	print(cmd)
	subprocess.run(cmd, shell=True)
for i in range(3):
	cmd =f'python detectyolov5powerinferall.py yolov5models/{edgetpumodels[1]}  256' 
	print(cmd)
	subprocess.run(cmd, shell=True)
	
for i in range(3):
	cmd =f'python detectyolov5powerinfership.py yolov5models/{edgetpumodels2[0]}  448' 
	print(cmd)
	subprocess.run(cmd, shell=True)
	
for i in range(3):
	cmd =f'python detectyolov5powerinfership.py yolov5models/{edgetpumodels2[1]}  448' 
	print(cmd)
	subprocess.run(cmd, shell=True)


edgetpumodels=["yolov5s-diorall-int8_edgetpu",
"yolov5n-diorship-int8_edgetpu",
"yolov5s-diorship-int8_edgetpu",
"yolov5n-diorall-int8_edgetpu"]




for model in yolov5shipmodels:
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
		
for model in yolov5allmodels:
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


for model in yolov8shipmodels:
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


for model in diorallmodels:
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


for model in yolov5allmodels:
	for i in range(3):
		cmd =f'python detectyolov5all.py yolov5models/{model} 0'
		elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
		with open(f'logfiletime/total-time{model}.txt','a') as f:
			f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")
for model in yolov5shipmodels:
	for i in range(3):
		cmd =f'python detectyolov5ship.py yolov5models/{model} 0'
		elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
		with open(f'logfiletime/total-time{model}.txt','a') as f:
			f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")


for model in yolov8shipmodels:
	for i in range(3):
		cmd =f'python detectyolov8ship.py yolov8diorship/{model} True'
		elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
		with open(f'logfiletime/total-time{model}.txt','a') as f:
			f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")

for model in diorallmodels:
	for i in range(3):
		cmd =f'python detectyolov8all.py yolov8diorall/{model} True'
		elapsed_wall_time, elapsed_cpu_time = record_time(cmd)
		with open(f'logfiletime/total-time{model}.txt','a') as f:
			f.write("total "+f"{elapsed_cpu_time:.6f} {elapsed_wall_time:.6f}\n")

################################

for model in yolov5allmodels:
	for i in range(3):
		cmd =f'psrecord "python detectyolov5all.py yolov5models/{model} 0" --log "processrecordings/"{model}{i}.txt --include-children' 
		print(cmd)
		subprocess.run(cmd, shell=True)
		
for model in yolov5shipmodels:
	for i in range(3):
		cmd =f'psrecord "python detectyolov5ship.py yolov5models/{model} 0" --log "processrecordings/"{model}{i}.txt --include-children' 
		print(cmd)
		subprocess.run(cmd, shell=True)



for model in yolov8shipmodels:
	for i in range(3):
		cmd =f'psrecord "python detectyolov8ship.py yolov8diorship/{model}" --log "processrecordings/"{model}{i}.txt --include-children' 
		print(cmd)
		subprocess.run(cmd, shell=True)


for model in diorallmodels:
	for i in range(3):
		cmd =f'psrecord "python detectyolov8all.py yolov8diorall/{model}" --log "processrecordings/"{model}{i}.txt --include-children' 
		print(cmd)
		subprocess.run(cmd, shell=True)		
		
################################################

edgetpumodels=["yolov5s-diorall-int8_edgetpu",
"yolov5n-diorship-int8_edgetpu",
"yolov5s-diorship-int8_edgetpu",
"yolov5n-diorall-int8_edgetpu"]




for model in yolov5shipmodels:
	for i in range(3):
		cmd =f'python detectyolov5powerinfership.py yolov5models/{model} 0'
		print(cmd)
		subprocess.run(cmd, shell=True)
		
for model in yolov5allmodels:
	for i in range(3):
		cmd =f'python detectyolov5powerinferall.py yolov5models/{model} 0'
		print(cmd)
		subprocess.run(cmd, shell=True)

for model in yolov8shipmodels:
	for i in range(3):
		cmd =f'python detectyolov8powerinfership.py yolov8diorship/{model}' 
		print(cmd)
		subprocess.run(cmd, shell=True)



for model in diorallmodels:
	for i in range(3):
		cmd =f'python detectyolov8powerinferall.py yolov8diorall/{model}' 
		print(cmd)
		subprocess.run(cmd, shell=True)		


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
