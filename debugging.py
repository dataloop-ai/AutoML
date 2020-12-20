import ray
# ray.init()

@ray.remote
def f(x):
    ray.util.pdb.set_trace()
    return x * x

futures = [f.remote(i) for i in range(2)]
print(ray.get(futures))