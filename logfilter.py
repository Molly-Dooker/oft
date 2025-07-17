import re
import csv
import os
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"폰트 설정 중 오류 발생: {e}. 기본 폰트로 진행합니다.")

def parse_log_to_csv(log_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_csv_path = os.path.join(output_dir, 'train.csv')
    val_csv_path = os.path.join(output_dir, 'validation.csv')

    train_data = []
    validation_data = []

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

def find_col(df, name):
    for col in df.columns:
        if col.strip().lower() == name.lower():
            return col
    return None

def save_plot(df:pd.DataFrame, title:str, output_dir:str):
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("알림: 'Malgun Gothic' 폰트를 찾을 수 없어 기본 폰트로 설정합니다.")    
    epoch_col = find_col(df, 'epoch')
    x_axis = df[epoch_col] if epoch_col else df.index
    score_col = find_col(df, 'score')
    total_col = find_col(df, 'total')
    other_columns = [
        col for col in df.columns 
        if col not in [epoch_col, score_col, total_col]
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()    
    current_ax_idx = 0
    if score_col or total_col:
        ax = axes[current_ax_idx]
        if score_col:
            ax.plot(x_axis, df[score_col], marker='o', linestyle='-', label=score_col)
        if total_col:
            ax.plot(x_axis, df[total_col], marker='s', linestyle='--', label=total_col)        
        ax.set_title('Score & Total per Epoch')
        ax.set_xlabel('Epoch' if epoch_col else 'Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        current_ax_idx += 1
    for col in other_columns:
        if current_ax_idx < len(axes):
            ax = axes[current_ax_idx]
            ax.plot(x_axis, df[col], marker='o', linestyle='-', label=col)
            ax.set_title(f'{col} per Epoch')
            ax.set_xlabel('Epoch' if epoch_col else 'Index')
            ax.set_ylabel(col)
            ax.legend()
            ax.grid(True)
            current_ax_idx += 1
    for i in range(current_ax_idx, len(axes)):
        fig.delaxes(axes[i])
    fig.suptitle(f'{title} Results Visualization', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(output_dir,f'{title}.png')
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"✅ 그래프가 '{filename}' 파일로 성공적으로 저장되었습니다.")

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
    
    train_df = pd.read_csv(os.path.join(args.output_dir, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(args.output_dir, 'validation.csv'))
    train_df.columns = train_df.columns.str.strip()
    valid_df.columns = valid_df.columns.str.strip()
    
    save_plot(train_df,'Training',args.output_dir)
    save_plot(valid_df,'Validation',args.output_dir)