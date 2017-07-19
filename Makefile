SERVER_KEY='/Users/tims/.ssh/terraform'
SERVER_IP='35.190.139.8'
SERVER_USER='core'

sync:
	rsync -e "ssh -i $(SERVER_KEY)" -rvL --exclude=.git . $(SERVER_USER)@$(SERVER_IP):~/tacotron

ssh:
	ssh -ti $(SERVER_KEY) $(SERVER_USER)@$(SERVER_IP) "cd tacotron && bash"

base-conda:
	curl https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh > /tmp/Anaconda3-4.4.0-Linux-x86_64.sh
	bash /tmp/Anaconda3-4.4.0-Linux-x86_64.sh
	conda install pytorch torchvision cuda80 -c soumith
	sudo apt-get install libav-tools

download:
	scp -r -i $(SERVER_KEY) $(SERVER_USER)@$(SERVER_IP):~/tacotron/demo.wav ./
