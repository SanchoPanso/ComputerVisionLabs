# Задание

1. Откалибровать собственную камеру при помощи доски aruco.

2. Сгенерировать маркер Aruco из понравившегося словаря

3. Написать программу, которая бы детектировала маркер на изображении и рисовать куб с основанием в виде маркера (куб должен быть спроецирован на плоскость изображения и иметь различные цвета ребер). Используйте захват видеопотока с камеры (VideoCapture), чтобы получать изображения.


# Отчет

## Задание 1

Для калибровки камеры модуле aruco существует файл calibrate_camera.cpp. Он был скомпилирован и запущен с параметрами:
```
calibrate_camera.exe -d=10 --dp=detector_params.yml -h=10 -w=7 -l=0.022 -s=0.0105 --zt=true --ci=1 calibration.xml
```

Файл `detector.yaml` содержит следующие параметры:
```
%YAML:1.0
adaptiveThreshWinSizeMin: 3
adaptiveThreshWinSizeMax: 23
adaptiveThreshWinSizeStep: 10
adaptiveThreshWinSize: 21
adaptiveThreshConstant: 7
minMarkerPerimeterRate: 0.03
maxMarkerPerimeterRate: 4.0
polygonalApproxAccuracyRate: 0.05
minCornerDistanceRate: 0.05
minDistanceToBorder: 3
minMarkerDistance: 10.0
minMarkerDistanceRate: 0.05
cornerRefinementMethod: 0
cornerRefinementWinSize: 5
cornerRefinementMaxIterations: 30
cornerRefinementMinAccuracy: 0.1
markerBorderBits: 1
perspectiveRemovePixelPerCell: 8
perspectiveRemoveIgnoredMarginPerCell: 0.13
maxErroneousBitsInBorderRate: 0.04
minOtsuStdDev: 5.0
errorCorrectionRate: 0.6

# new aruco 3 functionality
useAruco3Detection: 0
minSideLengthCanonicalImg: 32 # 16, 32, 64 --> tau_c from the paper
minMarkerLengthRatioOriginalImg: 0.02 # range [0,0.2] --> tau_i from the paper
cameraMotionSpeed: 0.1 # range [0,1) --> tau_s from the paper
useGlobalThreshold: 0
```

На выходе получился файл `calibration.xml` со следующим содержанием:

```
<?xml version="1.0"?>
<opencv_storage>
<calibration_time>"03/15/23 23:53:44"</calibration_time>
<image_width>640</image_width>
<image_height>480</image_height>
<flags>8</flags>
<camera_matrix type_id="opencv-matrix">
  <rows>3</rows>
  <cols>3</cols>
  <dt>d</dt>
  <data>
    6.7615594210621180e+02 0. 3.0706547817335900e+02 0.
    6.7131219777461365e+02 2.4133592000675327e+02 0. 0. 1.</data></camera_matrix>
<distortion_coefficients type_id="opencv-matrix">
  <rows>1</rows>
  <cols>5</cols>
  <dt>d</dt>
  <data>
    -3.0020531024597596e-01 1.5888856584296660e+00 0. 0.
    -3.4087951678482127e+00</data></distortion_coefficients>
<avg_reprojection_error>1.0303735229236632e+00</avg_reprojection_error>
</opencv_storage>
```

## Задание 2

Для создания маркера использована функция `generateImageMarker` в файле `create_aruco_marker.cpp`. Изображение
сохранено как `marker23.png`.
<div align="center">
  <img src="./marker23.png"/>
</div>

## Задание 3

В файле `main.cpp` реализован захват видео с камеры, детектирование маркеров, отрисовка кубов и запись полученного видео в файл.
В качестве маркера используется маркер, полученный в предыдущем задании.

Для обнаружения маркеров создан детектор маркеров для словаря DICT_6X6_250. Для каждого маркера с помощью функции `solvePnP`
и полученной в п.1 внутренней матрицы камеры
находится вектор перемещения и поворота (в форме Родрига) для преобразования из системы координат маркера в систему координат камеры. Из массива `cubePoints3d` берутся 3d точки куба, которые для каждого маркера, соединяясь, проектируются в линии на изображении с помощью функции `projectPoints`.

Полученное изображение выводится в окне и записывется в видео.
<div align="center">
  <img src="./cv_lab5.gif"/>
</div>
