import re
import csv
import os
from argparse import ArgumentParser

def parse_log_to_csv(log_file_path, output_dir):
    """
    주어진 로그 파일을 파싱하여 지정된 디렉터리에 train/validation 결과를 CSV로 저장합니다.
    """
    # 1. 결과 저장 디렉토리 생성 (없을 경우)
    # exist_ok=True 옵션은 디렉토리가 이미 존재해도 오류를 발생시키지 않습니다.
    os.makedirs(output_dir, exist_ok=True)

    # 2. 결과 CSV 파일의 전체 경로 설정
    # os.path.join을 사용하면 OS에 맞는 경로 구분자로 안전하게 경로를 합칠 수 있습니다.
    train_csv_path = os.path.join(output_dir, 'train.csv')
    val_csv_path = os.path.join(output_dir, 'validation.csv')

    # 데이터를 저장할 리스트 초기화
    train_data = []
    validation_data = []

    # 에포크 번호를 추출하기 위한 정규 표현식
    epoch_pattern = re.compile(r'=== epoch (\d+) of \d+ ===')

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ 오류: 로그 파일 '{log_file_path}'를 찾을 수 없습니다.")
        return

    current_epoch = 0
    # 로그 파일의 모든 라인을 순회
    for i, line in enumerate(lines):
        # 에포크 번호 감지 및 업데이트
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        # Training 결과 블록 처리
        if '==> Training complete' in line:
            if i + 5 < len(lines):
                results = {'epoch': current_epoch}
                for j in range(1, 6):
                    data_line = lines[i + j]
                    try:
                        key_part, value_part = data_line.split('|')[-1].split(':')
                        key = key_part.strip()
                        value = float(value_part.strip())
                        results[key] = value
                    except (ValueError, IndexError):
                        continue
                train_data.append(results)

        # Validation 결과 블록 처리
        elif '==> Validation complete' in line:
            if i + 5 < len(lines):
                results = {'epoch': current_epoch}
                for j in range(1, 6):
                    data_line = lines[i + j]
                    try:
                        key_part, value_part = data_line.split('|')[-1].split(':')
                        key = key_part.strip()
                        value = float(value_part.strip())
                        results[key] = value
                    except (ValueError, IndexError):
                        continue
                validation_data.append(results)

    # CSV 파일로 저장
    header = ['epoch', 'total', 'score', 'position', 'dimension', 'angle']

    if train_data:
        with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(train_data)
        print(f"✅ 학습 결과가 '{train_csv_path}' 파일에 성공적으로 저장되었습니다.")

    if validation_data:
        with open(val_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(validation_data)
        print(f"✅ 검증 결과가 '{val_csv_path}' 파일에 성공적으로 저장되었습니다.")


if __name__ == '__main__':
    parser = ArgumentParser(description="로그 파일을 파싱하여 Train/Validation 결과를 CSV로 저장합니다.")
    # 필수 인자: 로그 파일 경로
    parser.add_argument('-f', '--logpath', type=str, required=True,
                        help='파싱할 로그 파일의 경로')
    # 선택 인자: 결과를 저장할 디렉토리 경로
    parser.add_argument('-o', '--output_dir', type=str, default='.',
                        help='결과 CSV 파일을 저장할 디렉토리 (기본값: 현재 디렉토리)')

    args = parser.parse_args()
    parse_log_to_csv(args.logpath, args.output_dir)