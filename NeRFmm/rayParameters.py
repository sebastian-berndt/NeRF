class RayParameters():
    def __init__(self, near=0.0, far=1.0, n_sample=128, pos_enc_freq=10, dir_enc_freq=4):
      self.NEAR, self.FAR = near, far #0.0, 1.0  # ndc near far
      self.N_SAMPLE = n_sample #128  # samples per ray
      self.POS_ENC_FREQ = pos_enc_freq #10  # positional encoding freq for location
      self.DIR_ENC_FREQ = dir_enc_freq #4   # positional encoding freq for direction

