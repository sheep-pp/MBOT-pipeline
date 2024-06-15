import cv2
import tkinter as tk
from tkinter import simpledialog, filedialog

def ask_watch_video():
    root = tk.Tk()
    root.withdraw()
    answer = simpledialog.askstring("Watch video", "Do you want to see the video? (y/n):", parent=root)
    root.destroy()
    return answer.lower() == 'y'

def get_video_segment():
    root = tk.Tk()
    root.withdraw()
    start_time = simpledialog.askfloat("Start time", "Enter start time for the video segment (seconds):", parent=root)
    end_time = simpledialog.askfloat("End time", "Enter end time for the video segment (seconds):", parent=root)
    root.destroy()
    return start_time, end_time

def play_video_segment(video_path, start_time, end_time):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    while True: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) 
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # If no frame is returned, break from the loop
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_frame > end_frame:
                break  # If the frame exceeds end_frame, restart video

            cv2.imshow('Video', frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'): 
                cap.release()
                cv2.destroyAllWindows()
                return  

        # This part is reached if the video segment ends; it will then restart unless 'q' was pressed

    cap.release()
    cv2.destroyAllWindows()

def main():
    while ask_watch_video():  # allow asking if the user wants to watch a video
        start_time, end_time = get_video_segment()
        video_path = filedialog.askopenfilename(title="select video file")
        play_video_segment(video_path, start_time, end_time)

if __name__ == "__main__":
    main()