# 멀티프로세싱 + TCP 소켓 사용했을 때 RPC와 latency를 비교하기 위해 만든 테스트 스크립트
#
# 실행 방법:
# docker run -dt --net=host test python mp_resnet.py --port 1000; docker run -dt --net=host test python mp_resnet.py --port 1001; docker run -dt --net=host test python mp_resnet.py --port 1002
# docker run -dt --net=host test python mp_resnet.py --master True

from test import *
import argparse

num_workers = 3

# 마스터, 워커 모두 필요한 모델 만들어두기
orig = models.resnet50(pretrained = True)
split = SplitResnet(orig, num_workers)

def run_master_server():
    print('Started master server routine')
    worker_nodes = [WorkerNode(host = 'localhost', port = i + 1000) for i in range(num_workers)]
    distributed = DistributedResnet(split, worker_nodes, is_sequential = False)
    
    input_shape = [1, 3, 256, 256]
    assert_model_equality(orig, distributed, input_shape, num_tests = 5)

    distributed.analyze_overheads(input_shape, num_tests = 5, outer_tqdm_progress = None)

    print(f"parallel version distributed part total runtime: {distributed.runtime_record.net_runtime_per_category('partial convs'):.7f}")
    for i in range(len(worker_nodes)):
        print(f"worker node {i} total runtime: {distributed.runtime_record.net_runtime_per_category(f'worker {i}'):.7f}")

    # 모든 worker node 종료
    for worker_node in worker_nodes:
        tcp.send_utf8(worker_node.sock, 'terminate')

    print('Finished master server routine')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--master", type=bool, default=False)
    args.add_argument("--port", type=int, default=1000)
    args = args.parse_args()

    if args.master:
        run_master_server()
    else:
        test_worker_node_server(port = args.port, split = split)