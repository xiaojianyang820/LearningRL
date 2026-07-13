from buffers.off_policy_buffer import UniformOffPolicyBuffer


if __name__ == '__main__':
    buffer = UniformOffPolicyBuffer(obs_dim=4, act_dim=3, size=10)
    print(buffer)