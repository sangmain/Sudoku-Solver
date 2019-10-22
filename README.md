# Sudoku
sudoku solving program based on keras and opencv
라이브러리 opencv를 이용하여 스도쿠 이미지를 검출하고 검출된 스도쿠판의 각 cross점을 분석하여 셀을 분리, 분리된 셀에서 숫자가 있는지를 검사후 이진화하여 흑백의 이미지로 저장, 저장된 이미지는 tensorflow에서 숫자의 이미지를 softmax를 이용해 분류된 숫자를 이용해 스도쿠를 처리하는 프로젝트입니다.

인터넷에서 저장한 수도쿠 이미지
<img src="https://user-images.githubusercontent.com/26996823/67270618-fec40400-f4f3-11e9-972b-4b9c10f005db.png" width="90%"></img>

결과 이미지
<img src="https://user-images.githubusercontent.com/26996823/67270785-49de1700-f4f4-11e9-8768-37af64a38ff0.png" width="90%"></img>

