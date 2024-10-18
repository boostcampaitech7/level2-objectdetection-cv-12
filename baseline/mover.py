import os
import shutil

# 이미지 파일이 있는 폴더와 이동할 대상 폴더 경로
source_dir = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val/images'
destination_dir = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD/val'

# 파일 이동 함수
def move_files(src, dest):
    # 파일 목록 가져오기
    files = [f for f in os.listdir(src) if f.endswith('.jpg')]
    
    # 각 파일을 대상 폴더로 이동
    for file in files:
        src_file = os.path.join(src, file)
        dest_file = os.path.join(dest, file)
        shutil.move(src_file, dest_file)
        print(f"Moved {file} to {dest}")

# 파일 이동 실행
move_files(source_dir, destination_dir)
