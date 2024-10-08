
from socket import socket, error, AF_INET, SOCK_STREAM, MSG_WAITALL
import json
import warnings
import torch
import numpy as np
import traceback

def send_u32(sock: socket, value: int):
    sock.send(value.to_bytes(4, byteorder = 'little'))

def recv_u32(sock: socket) -> int:
    return int.from_bytes(sock.recv(4), byteorder = 'little')

def send_utf8(sock: socket, msg: str):
    encoded_msg = str.encode(msg)
    send_u32(sock, len(encoded_msg))
    sock.send(encoded_msg)

def recv_utf8(sock: socket) -> str:
    header_size = recv_u32(sock)
    return sock.recv(header_size).decode()

def send_json(sock: socket, obj):
    encoded_msg = json.dumps(obj).encode()
    send_u32(sock, len(encoded_msg))
    sock.send(encoded_msg)

def send_json_with_timestamp(sock: socket, obj):
    t1 = time.time()
    encoded_msg = json.dumps(obj).encode()
    t2 = time.time()
    send_u32(sock, len(encoded_msg))
    sock.send(encoded_msg)
    t3 = time.time()
    send_json(sock, {
        'total' : t3 - t1,
        'encode json' : t2 - t1,
        'send json' : t3 - t2
    })

def recv_json(sock: socket):
    header_size = recv_u32(sock)
    raw_bytes = sock.recv(header_size)
    result = json.loads(raw_bytes)
    return result

def recv_json_with_timestamp(sock: socket):
    t1 = time.time()
    header_size = recv_u32(sock)
    raw_bytes = sock.recv(header_size)
    t2 = time.time()
    result = json.loads(raw_bytes)
    t3 = time.time()

    timestamp = {
        'send' : recv_json(sock),
        'recv' : {
            'total' : t3 - t2,
            'receive data' : t2 - t1,
            'decoded json' : t3 - t2,
        }
    }
    return result, timestamp


def send_tensor(sock: socket, tensor: torch.Tensor):
    header = json.dumps({'shape': tensor.shape})
    assert(tensor.dtype == torch.float32)
    send_utf8(sock, header)
    
    encoded_tensor = tensor.numpy().tobytes()

    send_u32(sock, len(encoded_tensor))
    sock.send(encoded_tensor)

def send_tensor_with_timestamp(sock: socket, tensor: torch.Tensor):
    t1 = time.time()

    header = json.dumps({'shape': tensor.shape})
    assert(tensor.dtype == torch.float32)
    send_utf8(sock, header)

    t2 = time.time()
    
    encoded_tensor = tensor.numpy().tobytes()

    t3 = time.time()

    send_u32(sock, len(encoded_tensor))
    sock.send(encoded_tensor)

    t4 = time.time()

    send_json(sock, {
        'total' : t4 - t3,
        'send header' : t2 - t1,
        'encode tensor' : t3 - t2,
        'send data' : t4 - t3,
    })


def recv_tensor(sock: socket) -> torch.Tensor:
    header = recv_json(sock)
    tensor_size = recv_u32(sock)
    raw_bytes = sock.recv(tensor_size, MSG_WAITALL)
    result = torch.frombuffer(raw_bytes, dtype = torch.float32).reshape(header['shape'])

    return result

def recv_tensor_with_timestamp(sock: socket):
    t1 = time.time()

    header = recv_json(sock)

    t2 = time.time()

    tensor_size = recv_u32(sock)
    raw_bytes = sock.recv(tensor_size, MSG_WAITALL)

    t3 = time.time()

    result = torch.frombuffer(raw_bytes, dtype = torch.float32).reshape(header['shape'])

    t4 = time.time()

    timestamp = {
        'send' : recv_json(sock),
        'recv' : {
            'total' : t4 - t1,
            'receive header' : t2 - t1,
            'receive data' : t3 - t2,
            'reconstruct tensor' : t4 - t3,
        }
    }

    return result, timestamp

def create_server(host: str, port: int, backlog: int = 10):
    listen_sock = socket(AF_INET, SOCK_STREAM)

    try:
        listen_sock.bind((host, port))
        listen_sock.listen(backlog)
    except error as e:
        listen_sock.close()
        raise e
    
    return listen_sock

def connect_server(host: str, port: int):
    sock = socket(AF_INET, SOCK_STREAM)

    try:
        sock.connect((host, port))
    except error as e:
        sock.close()
        raise e
    
    return sock

def supress_immutable_tensor_warning():
    """
    recv_tensor() 함수를 사용할 때 immutable한 데이터로 mutable한 텐서를 만들었다며 표시되는 UserWarning을 숨겨줌.  
    우리는 해당 텐서를 수정하지 않고 추론에만 사용하므로 해당 경고를 무시해도 문제 없음!
    """
    warnings.filterwarnings('ignore', category = UserWarning)


from multiprocessing import Process
from tqdm import tqdm
import time

def test_tcp_server():
    try:
        with create_server('localhost', 9999) as listen_sock:
            client_sock, client_addr = listen_sock.accept()
            send_json_with_timestamp(client_sock, recv_json(client_sock))
            send_tensor_with_timestamp(client_sock, recv_tensor(client_sock))
            client_sock.close()
    except Exception as e:
        print('[Exception from test server]')
        traceback.print_exc()

def test_tcp_client():
    try:
        progress = tqdm(total = 3, desc = 'tcp communication assertion', postfix = 'connecting to server')
        with connect_server('localhost', 9999) as sock:
            progress.update()

            progress.set_postfix_str('json')
            sample_json = {'name' : 'haha', 'age' : 123}
            send_json(sock, sample_json)
            echo_json, timestamp = recv_json_with_timestamp(sock)
            assert(echo_json == sample_json)
            print('json echo timestamp', json.dumps(timestamp, indent = 4))
            time.sleep(1)
            progress.update()

            progress.set_postfix_str('tensor')
            sample_tensor = torch.rand(1, 3, 256, 256)
            send_tensor(sock, sample_tensor)
            echo_tensor, timestamp = recv_tensor_with_timestamp(sock)
            print('tensor echo timestamp', json.dumps(timestamp, indent = 4))
            assert(torch.equal(echo_tensor, sample_tensor))
            progress.update()
            progress.close()
    except Exception as e:
        print('[Exception from test client]')
        traceback.print_exc()

def assert_tcp_communication():
    supress_immutable_tensor_warning()
    progress = tqdm(total = 1, desc = 'initializing server')

    server = Process(target = test_tcp_server)
    server.start()

    time.sleep(1)
    progress.update()
    progress.close()

    client = Process(target = test_tcp_client)
    client.start()

    server.join()
    client.join()


if __name__ == '__main__':
    assert_tcp_communication()