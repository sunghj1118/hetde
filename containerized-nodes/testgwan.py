import socket
import torch
import time
from master import WorkerNode

# 워커 노드가 준비되었는지 확인하는 함수
def wait_for_worker(host: str, port: int, timeout: int = 60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"Worker {host}:{port} is ready.")
                return True
        except socket.error:
            print(f"Waiting for worker {host}:{port} to be ready...")
            time.sleep(5)  # 5초 기다린 후 다시 시도
    raise TimeoutError(f"Worker {host}:{port} is not ready after {timeout} seconds.")

# 테스트 함수에서 워커가 준비될 때까지 기다리기
def test_worker_communication():
    worker_nodes = ['worker1', 'worker2']
    ports = [1001, 1002]
    
    # 모든 워커 노드가 준비될 때까지 대기
    for worker, port in zip(worker_nodes, ports):
        wait_for_worker(worker, port)

    # 워커 노드와 통신 테스트
    nodes = [WorkerNode(worker, port) for worker, port in zip(worker_nodes, ports)]
    for node in nodes:
        test_tensor = torch.rand(1, 3, 224, 224)
        try:
            node.request_inference(test_tensor, "test_layer", 0)
            runtime, result = node.receive_inference_result()
            assert result is not None, f"Failed to receive response from worker: {node.sock.getpeername()}"
            print(f"Received response from worker {node.sock.getpeername()} successfully with runtime: {runtime}")
        except Exception as e:
            print(f"Error during communication with worker {node.sock.getpeername()}: {e}")

if __name__ == "__main__":
    test_worker_communication()
