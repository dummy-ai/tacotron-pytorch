from modules.prenet import *
from torch.autograd import Variable

def test_prenet():
	fc1_hidden_size = 256
	fc2_hidden_size = 128

	# simulate pre-net in decoder
	batch_size = 32
	input_size = 80
	input = Variable(torch.randn(batch_size, 1, input_size))

	prenet = PreNet(input_size,
					fc1_hidden_size=fc1_hidden_size,
					fc2_hidden_size=fc2_hidden_size)
	output = prenet(input)

	assert output.size() == (batch_size, 1, fc2_hidden_size)

	# simulate pre-net in encoder
	batch_size = 32
	embedding_size = 256
	time_steps = 17
	input2 = Variable(torch.randn(batch_size, time_steps, embedding_size))

	prenet2 = PreNet(embedding_size,
					 fc1_hidden_size=fc1_hidden_size,
					 fc2_hidden_size=fc2_hidden_size)

	output2 = prenet2(input2)

	assert output2.size() == (batch_size, time_steps, fc2_hidden_size)
