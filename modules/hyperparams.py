class Hyperparams:
    # data
    max_text_length = 20
    max_audio_length = 30

    # signal processing
    sr = 22050  # Sampling rate. Paper => 24000
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr*frame_shift)  # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length)  # samples This is dependent on the frame_length.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.2  # Exponent for amplifying the predicted magnitude
    n_iter = 30  # Number of inversion iterations
    use_log_magnitude = True  # if False, use magnitude

    # model/encoder
    embedding_dim = 256
    encoder_bank_k = 16
    encoder_bank_ck = 128
    encoder_proj_dims = (128, 128)
    encoder_highway_layers = 4
    encoder_highway_units = 128
    encoder_gru_units = 128

    # model/decoder
    attn_gru_hidden_size = 256
    decoder_gru_hidden_size = 256
    decoder_gru_layers = 2
    reduction_factor = 3  # 2, 3, 5 are used in paper

    # model/post
    post_bank_k = 8
    post_bank_ck = 128
    post_proj_dims = (256, 80)
    post_highway_layers = 4
    post_highway_units = 128

    # training scheme
    lr = 0.0001
    batch_size = 32
    num_epochs = 50000
    dropout = 0.5
    use_cuda = True
    teacher_forcing_ratio = 1.0
