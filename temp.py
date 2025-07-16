import os
from collections import Counter
from argparse import ArgumentParser

def count_kitti_labels(label_dir, start_index, end_index):
    """지정된 디렉토리의 KITTI 라벨 파일에서 클래스별 객체 수를 집계합니다."""
    
    # 디렉토리 존재 여부 확인
    if not os.path.isdir(label_dir):
        print(f"오류: '{label_dir}' 디렉토리를 찾을 수 없습니다.")
        return

    # 모든 클래스의 개수를 저장할 Counter 객체 생성
    total_counts = Counter()

    print(f"'{label_dir}' 디렉토리에서 파일 집계를 시작합니다...")
    print(f"대상 파일: {start_index:06d}.txt ~ {end_index:06d}.txt")

    # 지정된 범위의 모든 파일을 순회
    for i in range(start_index, end_index + 1):
        # 파일명을 6자리 숫자로 포맷팅 (예: 5 -> '000005.txt')
        file_name = f"{i:06d}.txt"
        file_path = os.path.join(label_dir, file_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 파일의 각 줄을 읽어와서 처리
                for line in f:
                    # 공백으로 줄을 나누고 첫 번째 단어(클래스명)를 가져옴
                    parts = line.strip().split()
                    if parts:
                        class_name = parts[0]
                        total_counts[class_name] += 1
        except FileNotFoundError:
            # 파일이 존재하지 않으면 건너뜀
            continue
        except IndexError:
            # 빈 줄이 있는 경우를 대비한 예외 처리
            continue
    
    if not total_counts:
        print("\n집계된 객체가 없습니다. 파일 범위나 경로를 확인해주세요.")
        return

    print("\n--- 집계 완료! ---")
    print("## 클래스별 최종 객체 수")

    # 가장 많이 등장한 순서대로 정렬하여 출력
    for class_name, count in total_counts.most_common():
        print(f"- {class_name}: {count}개")


if __name__ == '__main__':
    parser = ArgumentParser(description="디렉토리 내의 KITTI 라벨 파일에서 클래스별 객체 수를 집계합니다.")
    
    # 필수 인자: 라벨 파일이 있는 디렉토리 경로
    parser.add_argument('-d', '--dir', type=str, required=True, 
                        help='라벨 파일(e.g., 000000.txt)이 있는 디렉토리 경로')
    
    # 선택 인자: 파일 범위 지정
    parser.add_argument('--start', type=int, default=0, 
                        help='시작 파일 번호 (기본값: 0)')
    parser.add_argument('--end', type=int, default=7480, 
                        help='종료 파일 번호 (기본값: 7480)')

    args = parser.parse_args()
    
    count_kitti_labels(args.dir, args.start, args.end)