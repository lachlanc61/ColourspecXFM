import multiprocessing as mp

def worker(p):
    msg = 'Hello from child!'
    print(f"sending {msg} to parent")
    p.send(msg)
    v = p.recv()
    print(f"got {v} from parent")

if __name__ == '__main__':
    p_conn, c_conn = mp.Pipe()
    p = mp.Process(target=worker, args=(c_conn,))
    p.start()
    msg = 'Hello from parent!'
    print(f"got {p_conn.recv()} from child")
    print("sending {msg} to child")
    p_conn.send(msg)
    #p.join()