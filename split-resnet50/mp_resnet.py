# 멀티프로세싱 + TCP 소켓 사용했을 때 RPC와 latency를 비교하기 위해 만든 테스트 스크립트
#
# 실행 방법:
# docker run -dt -p 1000:29500 test python mp_resnet.py; docker run -dt -p 1001:29500 test python mp_resnet.py; docker run -dt -p 1002:29500 test python mp_resnet.py
# docker run -it --rm test python mp_resnet.py --master True
#
# 지금은 어째서인지 tcp.connect_server()는 성공하는데
# 워커 노드의 서버 쪽에서 sock.accept() 뒤로 진행이 안 되는 상황...
# netstat으로 소켓이 LISTEN 상태인거 확인했고,
# 마스터 서버에서 이상한 포트로 접속하려하면 에러 뜨는것도 확인했으니
# 에러가 없으면 뭔가 접속이 되어야하는데 왜? 안되??는거?지??

from test import *
import argparse

num_workers = 3

# 마스터, 워커 모두 필요한 모델 만들어두기
orig = models.resnet50(pretrained = True)
split = SplitResnet(orig, num_workers)

def run_master_server():
    print('Started master server routine')
    worker_nodes = [WorkerNode(host = 'host.docker.internal', port = i + 1000) for i in range(num_workers)]
    distributed = DistributedResnet(split, worker_nodes, is_sequential = False)
    
    input_shape = [1, 3, 256, 256]
    assert_model_equality(orig, distributed, input_shape, num_tests = 5)

    distributed.analyze_overheads(input_shape, num_tests = 5, outer_tqdm_progress = None)

    print(f"parallel version distributed part total runtime: {distributed.runtime_record.net_runtime_per_category('partial convs'):.7f}")
    for i in range(len(worker_nodes)):
        print(f"worker node {i} total runtime: {distributed.runtime_record.net_runtime_per_category(f'worker {i}'):.7f}")

    print('Finished master server routine')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--master", type=bool, default=False)
    args = args.parse_args()

    if args.master:
        run_master_server()
    else:
        test_worker_node_server(port = 29500, split = split, host = 'host.docker.internal')