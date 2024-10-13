
from socket import socket, error, AF_INET, SOCK_STREAM, MSG_WAITALL, SOL_SOCKET, SO_REUSEADDR
import json
import warnings
import torch
import traceback
import struct

def send_u32(sock: socket, value: int):
    sock.send(value.to_bytes(4, byteorder = 'little'))

def recv_u32(sock: socket) -> int:
    return int.from_bytes(sock.recv(4), byteorder = 'little')

def send_f64(sock: socket, value: float):
    sock.send(struct.pack('<d', value))

def recv_f64(sock: socket) -> float:
    result, = struct.unpack('<d', sock.recv(8))
    return result

def send_utf8(sock: socket, msg: str):
    encoded_msg = str.encode(msg)
    send_u32(sock, len(encoded_msg))
    sock.send(encoded_msg)

def recv_utf8(sock: socket) -> str:
    header_size = recv_u32(sock)
    return sock.recv(header_size).decode()

def send_json(sock: socket, obj, print_time: bool = False):
    t1 = time.time()
    encoded_msg = json.dumps(obj).encode()
    t2 = time.time()
    send_u32(sock, len(encoded_msg))
    sock.send(encoded_msg)
    t3 = time.time()
    if print_time:
        print('send_json: {:.7f} (encode json: {:.2f}%, send data: {:.2f}%)'.format(t3 - t1, (t2 - t1) / (t3 - t1) * 100, (t3 - t2) / (t3 - t1) * 100))

def recv_json(sock: socket, print_time: bool = False) -> str:
    t1 = time.time()
    header_size = recv_u32(sock)
    raw_bytes = sock.recv(header_size)
    t2 = time.time()
    result = json.loads(raw_bytes)
    t3 = time.time()

    if print_time:
        print('recv_json: {:.7f} (receive data: {:.2f}%, parse json: {:.2f}%)'.format(t3 - t1, (t2 - t1) / (t3 - t1) * 100, (t3 - t2) / (t3 - t1) * 100))
    return result

def send_tensor(sock: socket, tensor: torch.Tensor, print_time: bool = False):
    assert(tensor.dtype == torch.float32)

    t1 = time.time()

    send_u32(sock, len(tensor.shape))
    for dim in tensor.shape:
        send_u32(sock, dim)

    t2 = time.time()
    
    encoded_tensor = tensor.numpy().tobytes()

    t3 = time.time()

    send_u32(sock, len(encoded_tensor))
    sock.send(encoded_tensor)
    t4 = time.time()

    if print_time:
        print('send_tensor: {:.7f} (send header: {:.2f}%, encoded tensor: {:.2f}%, send data: {:.2f}%)'.format(t4 - t1, (t2 - t1) / (t4 - t1) * 100, (t3 - t2) / (t4 - t1) * 100, (t4 - t3) / (t4 - t1) * 100))

def recv_tensor(sock: socket, print_time: bool = False) -> torch.Tensor:
    t1 = time.time()

    shape_len = recv_u32(sock)
    shape = [recv_u32(sock) for _ in range(shape_len)]

    t2 = time.time()
    tensor_size = recv_u32(sock)
    raw_bytes = sock.recv(tensor_size, MSG_WAITALL)
    t3 = time.time()
    result = torch.frombuffer(raw_bytes, dtype = torch.float32).reshape(shape)
    t4 = time.time()

    if print_time:
        print('recv_tensor: {:.7f} (receive header: {:.2f}%, receive data: {:.2f}%, reconstruct tensor: {:.2f}%)'.format(t4 - t1, (t2 - t1) / (t4 - t1) * 100, (t3 - t2) / (t4 - t1) * 100, (t4 - t3) / (t4 - t1) * 100))

    return result

def create_server(host: str, port: int, backlog: int = 10):
    listen_sock = socket(AF_INET, SOCK_STREAM)
    listen_sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1) # close() 이후 발생하는 TIME_WAIT 상태 없애기

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


import threading
from tqdm import tqdm
import time

def test_tcp_server():
    try:
        with create_server('localhost', 9999) as listen_sock:
            client_sock, client_addr = listen_sock.accept()
            send_f64(client_sock, recv_f64(client_sock))
            send_json(client_sock, recv_json(client_sock))
            send_tensor(client_sock, recv_tensor(client_sock))
            client_sock.close()
    except Exception as e:
        print('[Exception from test server]')
        traceback.print_exc()

def test_tcp_client():
    try:
        progress = tqdm(total = 4, desc = 'tcp communication assertion', postfix = 'connecting to server')
        with connect_server('localhost', 9999) as sock:
            progress.update()

            progress.set_postfix_str('float')
            sample_float = 3.141592
            send_f64(sock, sample_float)
            assert(recv_f64(sock) == sample_float)
            time.sleep(0.1)
            progress.update()


            progress.set_postfix_str('json')
            sample_json = {'name' : 'haha', 'age' : 123}
            send_json(sock, sample_json, print_time = True)
            assert(recv_json(sock, print_time = True) == sample_json)
            time.sleep(0.1)
            progress.update()

            progress.set_postfix_str('tensor')
            sample_tensor = torch.rand(1, 3, 256, 256)
            send_tensor(sock, sample_tensor, print_time = True)
            assert(torch.equal(recv_tensor(sock, print_time = True), sample_tensor))
            progress.update()
            progress.close()
    except Exception as e:
        print('[Exception from test client]')
        traceback.print_exc()

def assert_tcp_communication():
    supress_immutable_tensor_warning()
    progress = tqdm(total = 1, desc = 'initializing server')

    server = threading.Thread(target = test_tcp_server)
    server.start()

    time.sleep(0.1)
    progress.update()
    progress.close()

    client = threading.Thread(target = test_tcp_client)
    client.start()

    server.join()
    client.join()



def test_tcp_server_listen_only():
    try:
        with create_server('localhost', 9999) as listen_sock:
            client_sock, client_addr = listen_sock.accept()
            send_u32(client_sock, recv_u32(client_sock))
            client_sock.close()
    except Exception as e:
        print('[Exception from test server]')
        traceback.print_exc()
    
def assert_immediate_socket_reuse():
    for i in tqdm(range(5), desc = 'immediate socket reuse assertion'):
        server = threading.Thread(target = test_tcp_server_listen_only)
        server.start()
        time.sleep(0.1)
        client = connect_server('localhost', 9999)

        send_u32(client, i)
        assert(recv_u32(client) == i)

        client.close()
        server.join()

if __name__ == '__main__':
    assert_tcp_communication()
    assert_immediate_socket_reuse()