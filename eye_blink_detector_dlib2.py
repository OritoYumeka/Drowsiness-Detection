import requests
#import tkinter
#import tkinter.font
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import time
import mediapipe as mp
from plyer import notification
import sys
import dlib
from imutils import face_utils
from scipy.spatial import distance
from pyicloud import PyiCloudService
import googlemaps
from geopy.distance import geodesic
import os
import ssl
import urllib3
#from PIL import Image, ImageTk  # Pillowライブラリを使って画像をTkinterに表示
#from tkinter import Label
#from tkinter import StringVar
import streamlit as st

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Global flags
start_flag = False
quitting_flag = False
count = 0
bad_count = 0
measurement_active = False
cap = None  # カメラのVideoCaptureオブジェクトを格納

# Google Maps API key
MAPS_API_KEY = 'AIzaSyCZkFgsU30wpK-PjIe7MKQgAZuLlhT3qrY'
gmaps = googlemaps.Client(key=MAPS_API_KEY)

device=None
api = None  # iCloud APIオブジェクト

def get_current_location():
    try:
        if device is None:
            print("デバイスが設定されていません。")
            return None, None
            
        location = device.location()
        if location:
            return location['latitude'], location['longitude']
        else:
            print("デバイスがオフラインか、位置情報が無効です。")
            return None, None
    except Exception as e:
        print(f"位置情報取得中にエラーが発生しました: {e}")
        return None, None
    
# カメラ映像を表示する関数
def show_frame():
    global cap, start_flag
    if start_flag:
        ret, frame = cap.read()  # カメラからフレームを取得
        if ret:
            # OpenCVのBGR形式をRGB形式に変換
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)  # フレームをPIL Imageに変換
            imgtk = ImageTk.PhotoImage(image=img)  # ImageをTkinterに表示できる形式に変換

            video_label.imgtk = imgtk  # 参照を保持しないと画像が消える
            video_label.config(image=imgtk)  # ラベルに画像を設定

        # 10ミリ秒後に再度show_frame関数を呼び出してフレームを更新
        video_label.after(10, show_frame)

def geocode(address):
    try:
        result = gmaps.geocode(address)
        if result:
            lat = result[0]['geometry']['location']['lat']
            lng = result[0]['geometry']['location']['lng']
            return lat, lng
        else:
            print(f"指定された住所から位置情報を取得できませんでした: {address}")
            return None, None
    except Exception as e:
        print(f"位置情報のジオコード中にエラーが発生しました: {e}")
        return None, None

def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)

def eye_marker(face_mat, position):
    for i, (x, y) in enumerate(position):
        cv2.circle(face_mat, (x, y), 1, (255, 255, 255), -1)
        cv2.putText(face_mat, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
count = 0
bad_count = 0

def download_file(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

def get_location_thread():
    global quitting_flag, start_flag, measurement_active
    while not quitting_flag:
        if start_flag:
            destination = dest_txt.get()
            dest_lat, dest_lng = geocode(destination)

            while not quitting_flag and start_flag and dest_lat is not None and dest_lng is not None:
                current_lat, current_lng = get_current_location()
                if current_lat is not None and current_lng is not None:
                    distance_to_dest = geodesic((current_lat, current_lng), (dest_lat, dest_lng)).kilometers
                    print(f"現在地と目的地の距離: {distance_to_dest:.2f} km")

def drowsiness_detection_thread():
    global start_flag, quitting_flag, count ,bad_count,measurement_active

     # Download cascade if not exists
    haarcascade_path = 'haarcascade_frontalface_alt2.xml'
    if not os.path.exists(haarcascade_path):
                print("Downloading Haar Cascade model...")
                download_file("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml", haarcascade_path)

    face_cascade = cv2.CascadeClassifier(haarcascade_path)

            # Download Dlib's face landmarks model if not exists
    shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(shape_predictor_path):
                print("Downloading Dlib shape predictor model...")
                download_file("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", shape_predictor_path)

                # Unzip the downloaded file
                import bz2
                with bz2.BZ2File(shape_predictor_path + '.bz2') as f_in, open(shape_predictor_path, 'wb') as f_out:
                    f_out.write(f_in.read())

    face_parts_detector = dlib.shape_predictor(shape_predictor_path)

    shoot_interval = shoot_txt.get()
    shoot_interval = int(shoot_interval)
    print(f"撮影間隔：{shoot_interval}")

    notice_time = notice_txt.get()
    notice_time = int(notice_time)
    print(f"眠気の検出回数：{notice_time}")

    run_time = run_txt.get()
    run_time = int(run_time) * 60
    print(f"測定時間：{run_time}")
    print(f"{run_time/shoot_interval}")

            #動画の読み込み
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
                print("Error: カメラにアクセスできません。")
                quit() # カメラが使えない場合はプログラムを終了

            # Get destination location
    destination = dest_txt.get()
    dest_lat, dest_lng = geocode(destination)
    if dest_lat is None or dest_lng is None:
        print("Error: 目的地が見つかりません。")
        quit()  # 目的地が設定できない場合もプログラムを終了

    count = 0  # カウンタの初期化

    while True:
        if start_flag:  # 開始フラグがTrueのときのみ実行
            ret, frame = cap.read()
        elif ret:#フレームが取得できたか
                print("Error: フレームが取得できません。")
                break  # フレームが取得できない場合はループを抜ける

        count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))

        if len(faces) == 1:#顔が検出されたら
            print("facesの検出")
            x, y, w, h = faces[0,:]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_gray = gray[y:(y + h), x:(x + w)]
            scale = 480 / h
            face_gray_resized = cv2.resize(face_gray, dsize=None, fx=scale, fy=scale)

            # 常に青い枠を表示
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            face = dlib.rectangle(0, 0, face_gray_resized.shape[1], face_gray_resized.shape[0])
            face_parts = face_parts_detector(face_gray_resized, face)
            face_parts = face_utils.shape_to_np(face_parts)

            left_eye = face_parts[42:48]
            eye_marker(face_gray_resized, left_eye)
            left_eye_ear = calc_ear(left_eye)

            right_eye = face_parts[36:42]
            eye_marker(face_gray_resized, right_eye)

            right_eye_ear = calc_ear(right_eye)
            print("left_eye>>>%f" % left_eye_ear)
            print("right_eye>>>%f" % right_eye_ear)
            print("count >>> %d" % count)

             # フラグの初期化
            sleepy_detected = False
            distance_calculated = False

            if count > 1:#count>1か？
                if (left_eye_ear + right_eye_ear) < 0.55:#左目と右目のEARが0.55未満か
                        bad_count += 1
                        print("bad count>>>%d" % bad_count)
                        print("眠そうな目を検出")
                        sleepy_detected = True

                # 現在地を取得し目的地までの距離を計算する
            current_lat, current_lng = get_current_location()
            if current_lat is not None and current_lng is not None:#現在地の緯度と経度は有効か？
                    distance_to_dest = geodesic((current_lat, current_lng), (dest_lat, dest_lng)).kilometers
                    print(f"現在地と目的地の距離: {distance_to_dest:.2f} km")

                    if distance_to_dest < 1:  # 目的地まで1km未満か？
                        distance_calculated = True

            # 両方の条件が満たされた場合にメッセージを表示
            if sleepy_detected and distance_calculated:
                    # 眠気が検出された場合にカメラ映像上にメッセージを表示
                    cv2.putText(frame, 
                                "Wake up!", 
                                (100, 200), # 表示位置
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4,  # フォントスケールを2に設定（大きさ）
                                (0, 0, 255), # 色は赤
                                10) # 太さを4に設定
           
        # OpenCVのBGR形式をRGB形式に変換してTkinterに表示
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    # 撮影の終了
    if count == (run_time / shoot_interval):
        print("時間になったので撮影終了します")
        cap.release()

def login_button_click():
    global device,api

    # iCloudのユーザー名、パスワード、デバイスIDの入力を取得
    username = username_txt.get()
    password = password_txt.get()

    try:
        api = PyiCloudService(username, password)

         # iCloudのデバイス一覧からdevice_idを探す
        print("デバイス一覧を取得:")
        devices = list(api.devices)
        device_options = [f"{dev['name']} ({dev['id']})" for dev in devices]

        if devices:
            # デバイスの選択肢をユーザーに提示して選ばせる
            device_selection.set(device_options[0])  # デフォルト選択を最初のデバイスに
            device_menu['menu'].delete(0, 'end')  # メニューをクリア
            for option in device_options:
                device_menu['menu'].add_command(label=option, command=tkinter._setit(device_selection, option))
            device_menu.config(state='normal')  # デバイス選択メニューを有効化
            confirm_button.config(state='normal')  # デバイス選択ボタンを有効化
        print(f"iCloud ログイン成功")
    except Exception as e:
        print(f"iCloud ログイン失敗: {e}")
        start_flag = False

def set_device():
                global device
                # 選択されたデバイスをセット
                selected_device = device_selection.get().split(" (")[1][:-1]  # デバイスIDだけを取得
                print(f"選択されたデバイスID: {selected_device}")
                 # Check if the API is initialized and devices are available
                if api is not None:
                    for dev in api.devices:
                        if dev['id'] == selected_device:
                            device = dev
                            print(f"使用デバイス: {device['name']}")
                            break
                    else:
                        print("指定されたデバイスIDが見つかりませんでした")
                        device = None
                else:
                    print("iCloud APIが初期化されていません")
                    device = None

                # Enable input fields after successfully setting the device
                if device is not None:
                    # ログイン成功後、撮影間隔等の入力フィールドを有効化
                    shoot_txt.config(state='normal')
                    notice_txt.config(state='normal')
                    run_txt.config(state='normal')
                    dest_txt.config(state='normal')

                    # Enable the start button
                    start_button.config(state='normal')

                    stop_button.config(state='disabled')
                    login_button.config(state='disabled')
                else:
                    start_button.config(state='disabled')

def start_button_click(event):
    global cap,start_flag, quitting_flag, measurement_active
    if not start_flag:
        start_flag = True
        quitting_flag = False
        cap = cv2.VideoCapture(0)  # カメラを起動
        if not cap.isOpened():
            print("Error: カメラにアクセスできません。")
            return
        t1 = threading.Thread(target=get_location_thread)
        t1.start()
        t2 = threading.Thread(target=drowsiness_detection_thread)
        t2.start()

        # スタートボタンを無効化し、ストップボタンを有効化
        start_button.config(state='disabled')
        stop_button.config(state='normal')

    # iCloudのユーザー名、パスワードの入力を取得して表示
    username = username_txt.get()
    password = password_txt.get()

    # 撮影間隔等の入力を取得
    shoot_interval = shoot_txt.get()
    notice_time = notice_txt.get()
    run_time = run_txt.get()
    destination = dest_txt.get()

def stop_button_click(event):
    global cap,start_flag
    start_flag = False
    quitting_flag = True
    if cap is not None:
        cap.release()  # カメラを解放
    video_label.config(image='')  # カメラ映像を消す

    print("一時停止ボタンが押下されました")

    # 一時停止ボタンを無効化し、スタートボタンを有効化
    stop_button.config(state='disabled')
    start_button.config(state='normal')
    
def quit_app():
    global quitting_flag,cap
    quitting_flag = True
    if cap is not None:
        cap.release()  # カメラを解放

    app.quit()

# Main window
app = tkinter.Tk()
app.title("sleepcare")
app.geometry("1000x1000")
app.configure(bg="pink")

# カメラ映像を表示するためのラベル
video_label = tkinter.Label(app)
video_label.pack(pady=20)

# UI elements
tkinter.Message(app, text="iCloudユーザー名", width=200, bg="#000fff000").pack(pady=10)
username_txt = tkinter.Entry(app)
username_txt.pack()

tkinter.Message(app, text="iCloudパスワード", width=200, bg="#000fff000").pack(pady=10)
password_txt = tkinter.Entry(app, show="*")
password_txt.pack()

login_button = tkinter.Button(app, text="ログイン", fg="blue", font=("Menlo", 20), command=login_button_click)
login_button.pack(pady=20)

# Device selection dropdown, initially disabled
device_selection = tkinter.StringVar(app)
device_selection.set("デバイスなし")  # 初期値
device_menu = tkinter.OptionMenu(app, device_selection, "デバイスなし")
device_menu.pack()
device_menu.config(state='disabled')  # デバイス選択は最初は無効

# Confirmation button for selecting the device, initially disabled
confirm_button = tkinter.Button(app, text="デバイス選択", command=set_device, state='disabled')
confirm_button.pack(pady=10)

tkinter.Message(app, text="撮影間隔(秒)", width=200, bg="#000fff000").pack(pady=10)
shoot_txt = tkinter.Entry(app)
shoot_txt.pack()
shoot_txt.insert(0, "1")  # デフォルト値として1秒

tkinter.Message(app, text="眠気の検出回数(回)", width=200, bg="#000fff000").pack(pady=10)
notice_txt = tkinter.Entry(app)
notice_txt.pack()
notice_txt.insert(0, "1")  # デフォルト値として1回

tkinter.Message(app, text="測定時間(分)", width=200, bg="#000fff000").pack(pady=10)
run_txt = tkinter.Entry(app)
run_txt.pack()
run_txt.insert(0, "1")  # デフォルト値として1分

tkinter.Message(app, text="目的地の地名", width=200, bg="#000fff000").pack(pady=10)
dest_txt = tkinter.Entry(app)
dest_txt.pack()

# Start and stop buttons
font = tkinter.font.Font(family="Helvetica", size=12, weight="bold")

start_button = tkinter.Button(app, text="スタート", fg="blue", font=("Menlo", 30))
start_button.place(x=250, y=650)
start_button.config(state='disabled')

stop_button = tkinter.Button(app, text="一時停止", fg="red", font=("Menlo", 30))
stop_button.place(x=425, y=650)

# Quit button
quit_button = tkinter.Button(app, text="終了", fg="black", font=("Menlo", 30), command=quit_app)
quit_button.place(x=600, y=650)

start_button.bind("<ButtonPress>", start_button_click)
stop_button.bind("<ButtonPress>", stop_button_click)
app.protocol("WM_DELETE_WINDOW", quit_app)

app.mainloop()
