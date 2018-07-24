import numpy as np  
import os
from timeit import default_timer as time

def write_test():
    

    took = []
    for bs in [32,64,128,256,512]:
        f = os.open('speed_test_file', os.O_CREAT|os.O_WRONLY, 0o777)
        # low-level I/O
        buff = os.urandom(bs << 20) # get random bytes
        start = time()

        os.write(f, buff)
        os.fsync(f) # force write to disk
        os.write(f, buff)
        os.fsync(f) # force write to disk
        os.write(f, buff)
        os.fsync(f) # force write to disk

        t = (time() - start)/3.
        took.append(bs/t)
        os.close(f)
        os.unlink('speed_test_file')
    print("Write speed: ({:.3f} +- {:.3f}) MB/s".format(np.mean(took),np.std(took)))
    return np.mean(took),np.std(took)

def compute_speed():
       
    N = 6000  
    M = 10000  

    k_list = [128, 256, 512, 1024, 2048]  

    def get_gflops(M, N, K):  
        return M*N*(2.0*K-1.0) / 1e9  

    #np.show_config() 

    timing = []
    gflops = []

    for K in k_list:  
        a = np.array(np.random.random((M, N)), dtype=np.double, order='C', copy=False)  
        b = np.array(np.random.random((N, K)), dtype=np.double, order='C', copy=False)  
        A = np.matrix(a, dtype=np.double, copy=False)  
        B = np.matrix(b, dtype=np.double, copy=False)  

        C = A*B  

        start = time()  

        C = A*B  
        C = A*B  
        C = A*B  
        C = A*B  
        C = A*B  

        end = time()  

        tm = (end-start) / 5.0  
        timing.append(tm)
        gflops.append(get_gflops(M, N, K) / tm)
    print("Compute speed: ({:.3f} +- {:.3f}) Gflop/s".format(np.mean(gflops),np.std(gflops)))
    return np.mean(gflops),np.std(gflops)

if __name__ == '__main__':
    write_test()
    compute_speed()
