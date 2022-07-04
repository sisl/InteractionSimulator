from intersim.envs import IntersimpleLidarFlat

def test_idm(tmp_path):
    env = IntersimpleLidarFlat(agent=42, idm=True)
    env.reset()
    env.render()

    for _ in range(1000):
        _, _, done, _ = env.step(-0.01)
        env.render()
        if done:
            break

    env.close(filestr=str(tmp_path))
