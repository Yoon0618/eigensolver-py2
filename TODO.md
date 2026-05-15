- params.json을 log.json 파일로 바꾸기
-> 파라미터 말고도 저장할 게 많음

- 각 함수별 소요 시간 기록해서 로그로 남기기

- 빠른 non-mode-파라미터 sweep을 위해서 mat_data 파일로 저장했다가
python run.py --load-mat mat1.npz 처럼 불러와서 진행하기
python run.py --save-mat mat1.npz 로 행렬 데이터 저장

- 긴 시간 적분 테스트를 위해 time evolution 최종 상태 불러와서 이어서 계산하기
python run.py --resume state1.npz

- RK4 말고 RK2로 했을 때 결과가 어떻게 달라지는지 비교하기