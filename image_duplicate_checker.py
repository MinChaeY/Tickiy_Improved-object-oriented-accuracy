from PIL import Image  # PIL 라이브러리에서 Image 모듈을 import  pip install pillow 필요함
import os  # os 모듈을 import
import hashlib  # hashlib 모듈을 import

def image_hash(filepath):
    """
    이미지 파일의 해시값을 반환하는 함수
    """
    # 파일을 바이너리 모드('rb')로 열고 'f'로 참조함
    with open(filepath, 'rb') as f:  
        # 파일을 읽어들임
        image_data = f.read()
        # 읽어들인 데이터에 대한 MD5 해시값을 반환  
        return hashlib.md5(image_data).hexdigest()  

def find_duplicate_images(directory):
    """디렉토리 내 중복 이미지를 찾는 함수"""
    hash_to_file = {} 
    duplicates = [] 

    for root, dirs, files in os.walk(directory): 
        for file in files:
            filepath = os.path.join(root, file) 
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):  # 파일이 이미지 파일인지 확인.(jpg만 쓰지만 혹시 모르니까)
                image_hash_value = image_hash(filepath)  # 이미지 파일의 해시값을 계산
                if image_hash_value in hash_to_file:  # 이미지 해시값이 이미 존재하는지 확인 해시값이 같을 경우엔 같은 이미지임
                    duplicates.append((filepath, hash_to_file[image_hash_value]))  # 중복된 이미지 경로를 리스트에 추가
                else:
                    hash_to_file[image_hash_value] = filepath  # 이미지 해시값과 파일 경로를 딕셔너리에 저장

    return duplicates  # 중복된 이미지들의 리스트를 반환

def rename_images(directory):
    """
    이미지 이름을 a1.jpg, a2.jpg, ... 형식으로 변경하는 함수
    """
    count = 1  # 이미지 이름을 변경할 때 사용할 숫자를 초기화
    for root, dirs, files in os.walk(directory): 
        for file in files: 
            filepath = os.path.join(root, file) 
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):  # 파일이 이미지 파일인지 확인
                new_filename = f"a{count}.jpg"  # 새로운 파일 이름을 설정
                os.rename(filepath, os.path.join(root, new_filename))  # 파일 이름을 변경함
                count += 1  # 숫자를 증가시켜 다음 파일 이름을 설정
    # 숫자 순서대로 변경하기
    count = 1  # 이미지 이름을 변경할 때 사용할 숫자를 초기화
    for root, dirs, files in os.walk(directory):  
        for file in files:  # 파일들을 순회
            filepath = os.path.join(root, file)  # 파일의 전체 경로를 가져옴
            if filepath.lower().endswith('.jpg'):  # 파일이 JPG 이미지인지 확인
                new_filename = f"M0{count}.jpg"  
                os.rename(filepath, os.path.join(root, new_filename))  # 파일 이름을 변경
                count += 1  # 숫자를 증가시켜 다음 파일 이름을 설정합니다.


if __name__ == "__main__":  # 이 스크립트가 직접 실행될 때만 실행되게 하기 위해 __name__으로 설정했음
    dpath = "G:\내 드라이브\Colab Notebooks\Tickie\Tickie_Improved-object-oriented-accuracy\image\layPeople"  # 이미지를 포함한 디렉토리의 경로를 설정합니다.
    duplicate_images = find_duplicate_images(dpath)  # 중복된 이미지를 찾음
    # 중복된 이미지가 있을 경우:
    if duplicate_images:  
        print("중복된 이미지를 발견했습니다:")  
        # 중복된 이미지 중 하나 삭제
        for dup in duplicate_images: 
            print(f"{dup[0]} 와 {dup[1]} 중 하나를 삭제합니다.") 
            os.remove(dup[0])  # 중복된 이미지 중 하나를 삭제
    else:  # 중복된 이미지가 없을 경우:
        print("중복된 이미지가 없습니다.")  

    # 이미지 이름을 숫자 순서대로 변경
    rename_images(dpath) 
