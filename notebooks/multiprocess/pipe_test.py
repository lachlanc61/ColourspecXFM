import multiprocessing as mp
import time

fn="/home/lachlan/CODEBASE/ReadoutXFM/notebooks/multiprocess/temp.txt"

def worker(fn, conn):
    with open(fn) as file:
        for line in file:
            conn.send(line)
            time.sleep(0.5)
        print("child finished")
        conn.close()   

if __name__ == '__main__':
    parent_conn, child_conn = mp.Pipe()
    process = mp.Process(target=worker, args=(fn, child_conn,))
    process.start()
    time.sleep(5)
    for i in range(10):
        print(f"{parent_conn.recv()}")
    print("RECEIVE FINISHED")
    process.join()
    print("PARENT FINISHED")