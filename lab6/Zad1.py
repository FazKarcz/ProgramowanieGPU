import tensorflow as tf
import time
import numpy as np

print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"GPU: {gpus}")
else:
    print("Słabo :/")


N = 4096

shape = (N, N)

mat_a = tf.random.normal(shape)
mat_b = tf.random.normal(shape)

def measure_multiplication(device_name, label):
    try:
        with tf.device(device_name):
            tf.matmul(tf.random.normal((100, 100)), tf.random.normal((100, 100)))

            start_time = time.time()
            c = tf.matmul(mat_a, mat_b)
            _ = c.numpy() 
            
            end_time = time.time()
            elapsed = end_time - start_time
            print(f" \n Czas wykonania {label}: {elapsed:.4f} s")
            return elapsed

    except RuntimeError as e:
        print(f"Błąd {label}: {e}")
        return None


# Pomiar na CPU
time_cpu = measure_multiplication('/CPU:0', 'CPU')

# omiar na GPU
if gpus:
    time_gpu = measure_multiplication('/GPU:0', 'GPU')
    
    if time_cpu and time_gpu:
        speedup = time_cpu / time_gpu
        print(f"\n GPU względem CPU: {speedup:.2f}x")



vector_size = 1_000_000

vec_a = tf.random.normal((vector_size,))
vec_b = tf.random.normal((vector_size,))

print(f"Obliczanie iloczynu skalarnego wektorów o rozmiarze {vector_size}...")

with tf.device('/GPU:0'):
    start_time = time.time()

    dot_product = tf.tensordot(vec_a, vec_b, axes=1)

    result_val = dot_product.numpy()
    end_time = time.time()

print(f"Wynik iloczynu skalarnego: {result_val:.4f}")
print(f"Czas obliczeń: {end_time - start_time:.6f} s")


M_SIZE = 10000

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    large_matrix = tf.random.normal((M_SIZE, M_SIZE))
    
    print("Obliczanie normy Frobeniusa...")
    start_time = time.time()
    
    frobenius_norm = tf.norm(large_matrix, ord='fro', axis=(0, 1))
    
    result_norm = frobenius_norm.numpy()
    end_time = time.time()

print(f"Norma Frobeniusa: {result_norm:.4f}")
print(f"Czas obliczeń: {end_time - start_time:.4f} s")
