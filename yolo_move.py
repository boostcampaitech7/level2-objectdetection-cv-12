import os

# txt 파일 경로 및 수정된 경로 저장할 파일 경로
input_txt = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD_YOLO/val/yolo_val.txt'  # 원본 txt 파일 경로
output_txt = '/data/ephemeral/home/level2-objectdetection-cv-12/FOLD_YOLO/val/yolo_val1.txt'  # 수정된 경로 저장할 파일 경로

# 파일 경로 수정
with open(input_txt, 'r') as infile, open(output_txt, 'w') as outfile:
    for line in infile:
        # 'images/images/' 부분을 'images/'로 변경
        fixed_line = line.replace('/images/images/', '/images/')
        outfile.write(fixed_line)

print(f'수정된 파일이 {output_txt}에 저장되었습니다.')
