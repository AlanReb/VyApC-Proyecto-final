import cv2
import numpy as np
import time
import threading
from collections import defaultdict, deque

# global listeners (pynput)
try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False

# optional active-window check (more fiable en Windows)
try:
    import pygetwindow as gw  # cross-platform wrapper; uses win32 on windows
    ACTIVE_WINDOW_AVAILABLE = True
except Exception:
    ACTIVE_WINDOW_AVAILABLE = False


class ExamMonitor:
    def __init__(self, camera_index=0, scale=0.6, exam_duration_sec=60 * 5):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara (index {}).".format(camera_index))

        # first frame read
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("No se pudo leer el primer frame de la cámara.")
        self.scale = scale
        self.frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        # selection & tracking (CAMShift)
        self.selection = None     # (x0,y0,x1,y1)
        self.drag_start = None
        self.tracking_state = 0
        self.track_window = None
        self.hist = None

        # exam control
        self.exam_running = False
        self.exam_duration_sec = exam_duration_sec
        self.exam_start_time = None
        self.exam_stop_time = None

        # statistics
        self.total_frames = 0
        self.attention_time = 0.0
        self.not_attention_time = 0.0

        # breakdown counters (seconds)
        self.not_attention_breakdown = defaultdict(float)  # keys: 'left','right','up','down','no_face','window_mouse','window_keyboard'
        self._last_frame_time = None

        # window change tracking
        self.window_change_events_mouse = 0
        self.window_change_events_keyboard = 0
        self.last_active_window = None
        self.active_window_check_interval = 1.0  # seconds

        # keyboard/mouse listener threads
        self._kb_listener = None
        self._mouse_listener = None

        # UI button geometry (in window coords)
        self.btn_rect = (10, 10, 140, 40)  # x,y,w,h
        self.btn_color_idle = (50, 180, 50)
        self.btn_color_running = (200, 60, 60)

        # umbrales de atencion (pixels)
        h, w = self.frame.shape[:2]
        self.center = (w // 2, h // 2)
        # if face center farther than these fractions -> consider turned
        self.x_frac_thresh = 0.18
        self.y_frac_thresh = 0.16

        # detector de rostros (Haar cascade) — robusto y ligero
        haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar)
        if self.face_cascade.empty():
            raise RuntimeError("No se pudo cargar el cascade de faces de OpenCV.")

        # window name and callback
        self.winname = "Monitor de examen"
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self.mouse_event)

        # lock
        self.lock = threading.Lock()

        # for smoothing face location over a few frames
        self.prev_face_centers = deque(maxlen=5)

        # prepare active window name if available
        if ACTIVE_WINDOW_AVAILABLE:
            try:
                self.last_active_window = gw.getActiveWindowTitle()
            except Exception:
                self.last_active_window = None

    # -------------------------
    # Vigilantes de eventos 
    # -------------------------
    def start_listeners(self):
        if PYNPUT_AVAILABLE:
            # keyboard listener: detect Alt+Tab, Win/Meta, or other combos
            def on_press(key):
                try:
                    # key could be e.g. Key.alt_l, Key.tab, Key.cmd, Key.cmd_r, Key.ctrl_l
                    kname = str(key)
                    if "Key.alt" in kname or "Key.tab" in kname or "Key.cmd" in kname or "Key.ctrl" in kname or "Key.windows" in kname:
                        # mark as keyboard window change event
                        with self.lock:
                            if self.exam_running:
                                self.window_change_events_keyboard += 1
                                # count 1 second (heuristic) as not-attention
                                self.not_attention_breakdown['cambio de ventana con teclado'] += 1.0
                except Exception:
                    pass

            def on_click(x, y, button, pressed):
                if pressed:
                    with self.lock:
                        if self.exam_running:
                            # heuristic: any global click might indicate window change
                            self.window_change_events_mouse += 1
                            self.not_attention_breakdown['cambio de ventana con mouse'] += 1.0

            self._kb_listener = keyboard.Listener(on_press=on_press)
            self._kb_listener.daemon = True
            self._kb_listener.start()

            self._mouse_listener = mouse.Listener(on_click=on_click)
            self._mouse_listener.daemon = True
            self._mouse_listener.start()

    def stop_listeners(self):
        try:
            if self._kb_listener:
                self._kb_listener.stop()
            if self._mouse_listener:
                self._mouse_listener.stop()
        except Exception:
            pass

    # -------------------------
    # mouse UI callback
    # -------------------------
    def mouse_event(self, event, x, y, flags, param):
        # scale incoming coordinates to window size (already scaled frames)
        # handle selection for tracking (drag)
        if event == cv2.EVENT_LBUTTONDOWN:
            # check if clicked on Start/Stop button
            bx, by, bw, bh = self.btn_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                # toggle exam
                if not self.exam_running:
                    self.start_exam()
                else:
                    self.stop_exam()
                return

            # else start selection (face ROI)
            self.drag_start = (x, y)
            self.selection = None
            self.tracking_state = 0

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            xo, yo = self.drag_start
            x0, y0 = np.minimum([xo, yo], [x, y])
            x1, y1 = np.maximum([xo, yo], [x, y])
            h, w = self.frame.shape[:2]
            x0, x1 = int(np.clip(x0, 0, w - 1)), int(np.clip(x1, 0, w - 1))
            y0, y1 = int(np.clip(y0, 0, h - 1)), int(np.clip(y1, 0, h - 1))
            if x1 - x0 > 5 and y1 - y0 > 5:
                self.selection = (x0, y0, x1, y1)

        elif event == cv2.EVENT_LBUTTONUP:
            # finalize selection
            self.drag_start = None
            if self.selection is not None:
                self.tracking_state = 1
                x0, y0, x1, y1 = self.selection
                self.track_window = (x0, y0, x1 - x0, y1 - y0)
                # compute histogram for CAMShift
                hsv_roi = cv2.cvtColor(self.frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                mask_roi = cv2.inRange(hsv_roi, np.array((0., 30., 10.)), np.array((180., 255., 255.)))
                hist = cv2.calcHist([hsv_roi], [0], mask_roi, [32], [0, 180])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                self.hist = hist.reshape(-1)

    # -------------------------
    # Examen
    # -------------------------
    def start_exam(self):
        with self.lock:
            self.exam_running = True
            self.exam_start_time = time.time()
            self.exam_stop_time = None
            self.total_frames = 0
            self.attention_time = 0.0
            self.not_attention_time = 0.0
            self.not_attention_breakdown = defaultdict(float)
            self.window_change_events_mouse = 0
            self.window_change_events_keyboard = 0
        print("[ExamMonitor] Examen iniciado el:", time.ctime(self.exam_start_time))

        # start listeners
        self.start_listeners()
        # if available, refresh last active window
        if ACTIVE_WINDOW_AVAILABLE:
            try:
                self.last_active_window = gw.getActiveWindowTitle()
            except Exception:
                self.last_active_window = None

    def stop_exam(self):
        with self.lock:
            self.exam_running = False
            self.exam_stop_time = time.time()
        self.stop_listeners()
        print("[ExamMonitor] Examen detenido el:", time.ctime(self.exam_stop_time))
        self.report()  # print results

    # -------------------------
    # Analisis de atención
    # -------------------------
    def analyze_attention(self, face_bbox):
        """
        face_bbox: (x,y,w,h) or None
        returns: (is_attending:bool, reason:string or None)
        """
        now = time.time()
        # measure dt from last frame
        if self._last_frame_time is None:
            dt = 0.0
        else:
            dt = now - self._last_frame_time
        self._last_frame_time = now
        if not self.exam_running:
            return True, None, dt

        if face_bbox is None:
            # No face detected => count as not-attention
            self.not_attention_breakdown['no hay rostro'] += dt
            self.not_attention_time += dt
            return False, 'no hay rostro', dt

        x, y, w, h = face_bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        # maintain smoothed center
        self.prev_face_centers.append((cx, cy))
        avg_cx = np.mean([c[0] for c in self.prev_face_centers])
        avg_cy = np.mean([c[1] for c in self.prev_face_centers])

        frame_h, frame_w = self.frame.shape[:2]
        dx = (avg_cx - frame_w / 2.0) / frame_w
        dy = (avg_cy - frame_h / 2.0) / frame_h

        # thresholds
        if abs(dx) <= self.x_frac_thresh and abs(dy) <= self.y_frac_thresh:
            # attending
            self.attention_time += dt
            return True, None, dt
        # else not attending: decide direction
        dir_label = None
        if dx < -self.x_frac_thresh:
            dir_label = 'viendo a la izquierda'
        elif dx > self.x_frac_thresh:
            dir_label = 'viendo a la derecha'
        elif dy < -self.y_frac_thresh:
            dir_label = 'viendo hacia arriba'
        elif dy > self.y_frac_thresh:
            dir_label = 'viendo hacia abajo'
        else:
            dir_label = 'desconocido'

        self.not_attention_breakdown[dir_label] += dt
        self.not_attention_time += dt
        return False, dir_label, dt

    # -------------------------
    # main loop
    # -------------------------
    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[ExamMonitor] Frame drop.")
                    time.sleep(0.01)
                    continue

                # resize & keep for mouse selection coords
                self.frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
                vis = self.frame.copy()

                # detect face if not tracking with CAMShift
                face_bbox = None
                if self.tracking_state == 1 and self.hist is not None and self.track_window is not None:
                    # CAMShift
                    hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                    prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                    # apply mask of valid colors
                    mask = cv2.inRange(hsv, np.array((0., 30., 10.)), np.array((180., 255., 255.)))
                    prob &= mask
                    # camshift
                    try:
                        track_box, self.track_window = cv2.CamShift(prob, tuple(self.track_window), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
                        pts = cv2.boxPoints(track_box).astype(int)
                        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
                        # approximate face bbox
                        x, y, w, h = self.track_window
                        face_bbox = (int(x), int(y), int(w), int(h))
                    except Exception:
                        # fallback to detection
                        self.tracking_state = 0

                if self.tracking_state == 0:
                    gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    if len(faces) > 0:
                        # choose largest face
                        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
                        (x, y, w, h) = faces[0]
                        face_bbox = (int(x), int(y), int(w), int(h))
                        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # analyze attention if exam running
                attending, reason, dt = self.analyze_attention(face_bbox)
                # draw small status text
                status_text = "RUNNING" if self.exam_running else "IDLE"
                # draw button
                bx, by, bw, bh = self.btn_rect
                color = self.btn_color_running if self.exam_running else self.btn_color_idle
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, -1)
                label = "Stop Exam" if self.exam_running else "Start Exam"
                cv2.putText(vis, label, (bx + 8, by + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # draw timer if running
                if self.exam_running:
                    elapsed = time.time() - self.exam_start_time
                    remaining = max(0, self.exam_duration_sec - elapsed)
                    mins = int(remaining // 60)
                    secs = int(remaining % 60)
                    cv2.putText(vis, f"Time left: {mins:02d}:{secs:02d}", (bx + bw + 10, by + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # check for end of exam
                    if elapsed >= self.exam_duration_sec:
                        self.stop_exam()

                # draw attention indicator
                if self.exam_running:
                    if attending:
                        cv2.putText(vis, "Poniendo atencion", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 240, 0), 2)
                    else:
                        cv2.putText(vis, f"No pone atencion: {reason}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # optionally draw current active window title
                if ACTIVE_WINDOW_AVAILABLE:
                    try:
                        cur = gw.getActiveWindowTitle()
                        cv2.putText(vis, f"Ventana activa: {cur[:40]}", (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                        # if changed and exam running, count
                        with self.lock:
                            if self.exam_running and self.last_active_window is not None and cur != self.last_active_window:
                                self.not_attention_breakdown['window_keyboard'] += 1.0
                                self.window_change_events_keyboard += 1
                            self.last_active_window = cur
                    except Exception:
                        pass

                # show image
                cv2.imshow(self.winname, vis)
                self.total_frames += 1

                # handle keys: ESC to quit, S to start/stop (alternative)
                key = cv2.waitKey(5) & 0xFF
                if key == 27:
                    # exit app (stop exam if running)
                    if self.exam_running:
                        self.stop_exam()
                    break
                if key == ord('s'):
                    # quick toggle
                    if self.exam_running:
                        self.stop_exam()
                    else:
                        self.start_exam()

            # end while
        finally:
            self.cleanup()

    def cleanup(self):
        self.stop_listeners()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    # -------------------------
    # reporting
    # -------------------------
    def report(self):
        duration = (self.exam_stop_time or time.time()) - (self.exam_start_time or time.time())
        attend = self.attention_time
        not_attend = self.not_attention_time
        pct_not = (not_attend / duration * 100) if duration > 0 else 0.0

        print("\n===== REPORTE TOTAL =====")
        print(f"Duracin: {duration:.1f} s")
        print(f"Tiempo total de atencion al examen: {attend:.1f} s")
        print(f"Tiempo sin poner atencion: {not_attend:.1f} s ({pct_not:.1f}%)")
        print("Resumen (en segundos):")
        for k, v in dict(self.not_attention_breakdown).items():
            print(f"  {k}: {v:.1f}s")
        print(f"Eventos de cambio de ventana (mouse): {self.window_change_events_mouse}")
        print(f"Eventos de cambio de ventana (teclado): {self.window_change_events_keyboard}")
        suspicious = pct_not > 40.0
        print(f"El comportamiento es sospechoso?? (>40% del tiempo sin poner atención): {suspicious}")
        print("=======================\n")


if __name__ == "__main__":
    # Examen de 3 minutos
    monitor = ExamMonitor(camera_index=0, scale=0.8, exam_duration_sec=180)
    monitor.run()
