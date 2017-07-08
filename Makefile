SERVER_KEY='/Users/tims/.ssh/terraform'
SERVER_IP='104.196.105.30'
SERVER_USER='core'

sync:
	rsync -e "ssh -i $(SERVER_KEY)" -rvL --exclude=.git . $(SERVER_USER)@$(SERVER_IP):~/tacotron

ssh:
	ssh -ti $(SERVER_KEY) $(SERVER_USER)@$(SERVER_IP) "cd tacotron && bash"
	

