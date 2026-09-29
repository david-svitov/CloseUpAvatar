
from websockets.sync.server import serve, ServerConnection
from threading import Thread
import threading
import time
from queue import Queue
from collections import deque

Q_out = Queue(1000)
Q_in = deque(maxlen=2)
clients = set()

def new_client(websocket: ServerConnection):
    global Q_out, Q_in
    if len(clients) != 0: 
        websocket.close()
        print(f"Reject new client")
        return
    
    Q_out = Queue(1000)
    Q_in.clear()
    clients.add(websocket)
    print(f"New client connected")

def client_left(websocket):
    clients.clear()
    print(f"Client disconnected")

def keep_send(websocket: ServerConnection):
    global Q_out
    try:
        while True:
            data = Q_out.get(block=True)
            websocket.send(data)
    except Exception as e:
        pass

def handler(websocket: ServerConnection):
    global Q_in
    new_client(websocket)
    Thread(target=keep_send, args=(websocket,), daemon=True).start()
    try:
        while True:
            message = websocket.recv()
            Q_in.append(message)
    except Exception as e:
        print(e)
        client_left(websocket)
    
def run_server(ip, port):
    with serve(handler, ip, port, compression=None, server_header=None) as server:
        print(f'WebSocket server is running on ws://{ip}:{port}')
        server.serve_forever()

#---------------------------------

def net_init(ip, port):
    Thread(target=run_server, args=(ip, port), daemon=True).start()

def recv():
    global Q_in
    try:
        data = Q_in.popleft()
    except IndexError:
        data = None
    return data

def send(data):
    global Q_out
    Q_out.put(data, block=True)

def is_connected():
    return len(clients) > 0

def wait_connection():
    status = 'alive'
    while True:
        if is_connected():
            return status
        status = 'connected'
        time.sleep(0.1)

##################

def main():
    net_init('127.0.0.1', 12346)
    while True:
        data = recv()
        if data is None: continue
        message = data
        send(message)

if __name__ == "__main__":
    main()
