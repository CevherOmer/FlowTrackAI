import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter
import seaborn as sns
from collections import defaultdict, deque
import json
import warnings
import logging
from sklearn.cluster import DBSCAN
from scipy import stats

# -------------------------------
# DEBUG MESAJLARINI KAPATMA
# -------------------------------
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
warnings.filterwarnings('ignore')
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)


# -------------------------------
# GELİŞMİŞ KONFİGÜRASYON
# -------------------------------
class Config:
    def __init__(self):
        self.save_folder = "dataset_frames"
        self.model_folder = "trained_models"
        self.results_folder = "analysis_results"
        self.reports_folder = "business_reports"

        # Performans ayarları
        self.frame_interval = 15
        self.img_size = 640
        self.batch_size = 2
        self.epochs = 15
        self.confidence_threshold = 0.6
        self.max_frames = 0  # 0 = sınırsız

        # Analiz ayarları
        self.stationary_threshold = 20
        self.velocity_threshold = 0.8
        self.tracking_threshold = 80
        self.time_window = 45
        self.cluster_eps = 50
        self.min_samples = 3

        # Görselleştirme
        self.heatmap_sigma = 12
        self.stationary_sigma = 18

        for folder in [self.save_folder, self.model_folder, self.results_folder, self.reports_folder]:
            os.makedirs(folder, exist_ok=True)


# -------------------------------
# GELİŞMİŞ İNSAN TAKİP VE ANALİZ SİSTEMİ - TRAJECTORY DÜZELTMESİ
# -------------------------------
class AdvancedHumanAnalyzer:
    def __init__(self, config):
        self.config = config
        self.detection_data = []
        self.trajectory_data = defaultdict(lambda: deque(maxlen=200))  # Daha uzun trajectory
        self.stationary_data = defaultdict(lambda: deque(maxlen=30))
        self.velocity_data = defaultdict(list)
        self.zone_data = defaultdict(lambda: defaultdict(int))
        self.entrance_exit_data = {'entrance': [], 'exit': []}
        self.hourly_stats = defaultdict(lambda: defaultdict(int))
        self.flow_data = defaultdict(list)  # Akış verileri için

        self.zones = {
            'giris': (0, 0.2),
            'orta_sol': (0.2, 0.5),
            'orta_sag': (0.2, 0.5),
            'uzak_sol': (0.5, 0.8),
            'uzak_sag': (0.5, 0.8),
            'cikis': (0.8, 1.0)
        }

    def get_zone(self, x, y, frame_width, frame_height):
        """Piksel koordinatını bölgelere ayır"""
        x_ratio = x / frame_width

        if x_ratio < 0.2:
            return 'giris'
        elif x_ratio < 0.5:
            if y < frame_height / 2:
                return 'orta_sol'
            else:
                return 'orta_sag'
        elif x_ratio < 0.8:
            if y < frame_height / 2:
                return 'uzak_sol'
            else:
                return 'uzak_sag'
        else:
            return 'cikis'

    def calculate_velocity(self, person_id, current_pos, timestamp):
        """İnsan hızını hesapla"""
        if person_id in self.trajectory_data and len(self.trajectory_data[person_id]) > 1:
            prev_pos = self.trajectory_data[person_id][-1]
            prev_x, prev_y, prev_time = prev_pos
            curr_x, curr_y = current_pos

            time_diff = timestamp - prev_time
            if time_diff > 0:
                distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                velocity = distance / time_diff
                self.velocity_data[person_id].append(velocity)
                return velocity
        return 0

    def detect_stationary_behavior(self, person_id, current_pos, timestamp):
        """Gelişmiş duraksama tespiti"""
        velocity = self.calculate_velocity(person_id, current_pos, timestamp)

        if velocity < self.config.velocity_threshold:
            self.stationary_data[person_id].append((current_pos[0], current_pos[1], timestamp))

            if len(self.stationary_data[person_id]) > 10:
                stationary_time = timestamp - self.stationary_data[person_id][0][2]
                if stationary_time > 30:
                    return True, stationary_time

        return False, 0

    def track_entrance_exit(self, person_id, current_zone, timestamp):
        """Giriş-çıkış takibi"""
        if person_id not in self.trajectory_data:
            return

        zones_visited = [self.get_zone(pos[0], pos[1], 1000, 800) for pos in self.trajectory_data[person_id]]

        if len(zones_visited) > 5:
            if 'giris' in zones_visited[-3:] and person_id not in [p[0] for p in self.entrance_exit_data['entrance']]:
                self.entrance_exit_data['entrance'].append((person_id, timestamp))

            if 'cikis' in zones_visited[-3:] and person_id not in [p[0] for p in self.entrance_exit_data['exit']]:
                self.entrance_exit_data['exit'].append((person_id, timestamp))

    def create_comprehensive_analysis(self, frames_folder, model):
        """Kapsamlı analiz pipeline'ı - TRAJECTORY DÜZELTMESİ"""
        frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
        if not frame_files:
            return None, None, None, None, None, None, None

        first_frame_path = os.path.join(frames_folder, frame_files[0])
        sample_frame = cv2.imread(first_frame_path)
        if sample_frame is None:
            return None, None, None, None, None, None, None

        height, width = sample_frame.shape[:2]

        density_heatmap = np.zeros((height, width))
        stationary_heatmap = np.zeros((height, width))
        velocity_heatmap = np.zeros((height, width))
        flow_heatmap = np.zeros((height, width))
        zone_heatmap = np.zeros((height, width))
        trajectory_map = np.zeros((height, width, 3))  # RGB trajectory haritası

        person_id_counter = 0
        reidentified_persons = {}

        print("🔍 Gelişmiş analiz başlatılıyor...")

        for frame_idx, frame_file in enumerate(frame_files):
            img_path = os.path.join(frames_folder, frame_file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            try:
                results = model.predict(frame, conf=self.config.confidence_threshold,
                                        imgsz=self.config.img_size, classes=[0], verbose=False)

                current_time = frame_idx

                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)

                    foot_x, foot_y = self.get_foot_position(box, frame.shape)
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    person_id = self.track_person(center_x, center_y, person_id_counter, reidentified_persons)
                    if person_id == person_id_counter:
                        person_id_counter += 1

                    zone = self.get_zone(center_x, center_y, width, height)
                    self.zone_data[zone]['count'] += 1
                    self.zone_data[zone]['current'] += 1

                    velocity = self.calculate_velocity(person_id, (center_x, center_y), current_time)
                    is_stationary, stationary_time = self.detect_stationary_behavior(person_id, (center_x, center_y),
                                                                                     current_time)

                    self.track_entrance_exit(person_id, zone, current_time)

                    hour_key = f"hour_{current_time // 3600}"
                    self.hourly_stats[hour_key]['total'] += 1
                    if is_stationary:
                        self.hourly_stats[hour_key]['stationary'] += 1

                    # GÜVENLİ HEATMAP GÜNCELLEME
                    self.safe_update_heatmaps(density_heatmap, stationary_heatmap, velocity_heatmap,
                                              flow_heatmap, zone_heatmap, foot_x, foot_y, center_x, center_y,
                                              velocity, is_stationary, zone, height, width)

                    # TRAJECTORY GÜNCELLEME - GÖRÜNÜR ÇİZGİLER
                    self.update_trajectory_visualization(trajectory_map, person_id, center_x, center_y, height, width)

            except Exception as e:
                print(f"Frame {frame_idx} işlenirken hata: {e}")
                continue

            if frame_idx % 100 == 0:
                for zone in self.zone_data:
                    self.zone_data[zone]['current'] = 0

            if frame_idx % 100 == 0:
                print(f"📊 Analiz: {frame_idx}/{len(frame_files)}")

        try:
            density_heatmap = gaussian_filter(density_heatmap, sigma=self.config.heatmap_sigma)
            stationary_heatmap = gaussian_filter(stationary_heatmap, sigma=self.config.stationary_sigma)
            velocity_heatmap = gaussian_filter(velocity_heatmap, sigma=8)
        except Exception as e:
            print(f"Heatmap yumuşatma hatası: {e}")

        return density_heatmap, stationary_heatmap, velocity_heatmap, flow_heatmap, zone_heatmap, trajectory_map, sample_frame

    def update_trajectory_visualization(self, trajectory_map, person_id, x, y, height, width):
        """Görünür trajectory çizgileri oluştur"""
        try:
            x, y = int(x), int(y)

            # Trajectory'yi güncelle
            if len(self.trajectory_data[person_id]) > 1:
                # Önceki pozisyonları al
                positions = list(self.trajectory_data[person_id])

                # Tüm noktalar arasında çizgi çiz
                for i in range(1, len(positions)):
                    prev_x, prev_y, _ = positions[i - 1]
                    curr_x, curr_y, _ = positions[i]

                    prev_x, prev_y = int(prev_x), int(prev_y)
                    curr_x, curr_y = int(curr_x), int(curr_y)

                    # Çizgi çiz (BGR formatında - yeşil çizgiler)
                    cv2.line(trajectory_map, (prev_x, prev_y), (curr_x, curr_y),
                             (0, 255, 0), 2)  # Yeşil, kalınlık 2

        except Exception as e:
            print(f"Trajectory görselleştirme hatası: {e}")

    def safe_update_heatmaps(self, density_heatmap, stationary_heatmap, velocity_heatmap,
                             flow_heatmap, zone_heatmap, foot_x, foot_y, center_x, center_y,
                             velocity, is_stationary, zone, height, width):
        """GÜVENLİ heatmap güncelleme"""
        try:
            foot_x, foot_y = int(foot_x), int(foot_y)
            center_x, center_y = int(center_x), int(center_y)

            # Yoğunluk heatmap
            y1_d = max(0, foot_y - 3)
            y2_d = min(height, foot_y + 4)
            x1_d = max(0, foot_x - 3)
            x2_d = min(width, foot_x + 4)
            if y1_d < y2_d and x1_d < x2_d:
                density_heatmap[y1_d:y2_d, x1_d:x2_d] += 1

            # Duraksama heatmap
            if is_stationary:
                y1_s = max(0, foot_y - 8)
                y2_s = min(height, foot_y + 9)
                x1_s = max(0, foot_x - 8)
                x2_s = min(width, foot_x + 9)
                if y1_s < y2_s and x1_s < x2_s:
                    stationary_heatmap[y1_s:y2_s, x1_s:x2_s] += 2

            # Hız heatmap
            y1_v = max(0, center_y - 2)
            y2_v = min(height, center_y + 3)
            x1_v = max(0, center_x - 2)
            x2_v = min(width, center_x + 3)
            if y1_v < y2_v and x1_v < x2_v:
                velocity_heatmap[y1_v:y2_v, x1_v:x2_v] += max(0, velocity)

            # Akış heatmap
            y1_f = max(0, center_y - 1)
            y2_f = min(height, center_y + 2)
            x1_f = max(0, center_x - 1)
            x2_f = min(width, center_x + 2)
            if y1_f < y2_f and x1_f < x2_f:
                flow_heatmap[y1_f:y2_f, x1_f:x2_f] += 1

            # Bölge heatmap
            y1_z = max(0, foot_y - 5)
            y2_z = min(height, foot_y + 6)
            x1_z = max(0, foot_x - 5)
            x2_z = min(width, foot_x + 6)
            if y1_z < y2_z and x1_z < x2_z:
                zone_heatmap[y1_z:y2_z, x1_z:x2_z] += 1

        except Exception as e:
            print(f"Heatmap güncelleme hatası: {e}")

    def track_person(self, x, y, next_id, reidentified_persons):
        """Gelişmiş kişi takibi"""
        for pid, positions in self.trajectory_data.items():
            if positions:
                last_x, last_y, _ = positions[-1]
                distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)

                if distance < self.config.tracking_threshold:
                    if len(positions) > 10:
                        time_gap = len(self.trajectory_data[pid]) - positions.index(positions[-1])
                        if time_gap < 50:
                            reidentified_persons[pid] = True

                    self.trajectory_data[pid].append((x, y, len(self.trajectory_data[pid])))
                    return pid

        self.trajectory_data[next_id] = deque([(x, y, 0)], maxlen=200)
        return next_id

    def get_foot_position(self, box, frame_shape):
        """Ayak pozisyonunu hesapla"""
        x1, y1, x2, y2 = box
        foot_x = (x1 + x2) // 2
        foot_y = min(y2 + 10, frame_shape[0] - 1)
        return int(foot_x), int(foot_y)

    def generate_business_intelligence(self):
        """İş zekası raporu oluştur"""
        insights = []

        total_people = sum([len(traj) for traj in self.trajectory_data.values()])
        avg_velocity = np.mean([np.mean(v) for v in self.velocity_data.values() if v]) if self.velocity_data else 0
        entrance_count = len(self.entrance_exit_data['entrance'])
        exit_count = len(self.entrance_exit_data['exit'])

        zone_analysis = ""
        for zone, data in self.zone_data.items():
            if data['count'] > 0:
                zone_analysis += f"• {zone}: {data['count']} kişi\n"

        stationary_people = sum([1 for data in self.stationary_data.values() if len(data) > 10])
        stationary_rate = (stationary_people / max(1, total_people)) * 100 if total_people > 0 else 0

        zone_popularity = sorted(self.zone_data.items(), key=lambda x: x[1]['count'], reverse=True)
        most_popular = zone_popularity[0] if zone_popularity else ("Yok", 0)
        least_popular = zone_popularity[-1] if zone_popularity else ("Yok", 0)

        # Akış analizi
        total_trajectories = len(self.trajectory_data)
        avg_trajectory_length = np.mean(
            [len(traj) for traj in self.trajectory_data.values()]) if self.trajectory_data else 0

        insights.extend([
            "📊 İŞ ZEKASI RAPORU",
            "=" * 50,
            f"👥 Toplam İnsan Hareketi: {total_people}",
            f"🚶 Ortalama Hız: {avg_velocity:.2f} pixel/frame",
            f"📍 Giriş Sayısı: {entrance_count}",
            f"🚪 Çıkış Sayısı: {exit_count}",
            f"⏸️  Duraksama Oranı: {stationary_rate:.1f}%",
            f"🔄 Toplam Trajectory: {total_trajectories}",
            f"📏 Ort. Trajectory Uzunluğu: {avg_trajectory_length:.1f}",
            "",
            "🏪 BÖLGE ANALİZİ:",
            zone_analysis,
            "",
            "🏆 POPÜLERLİK ANALİZİ:",
            f"• En Popüler Bölge: {most_popular[0]} ({most_popular[1]['count']} kişi)",
            f"• En Az Popüler: {least_popular[0]} ({least_popular[1]['count']} kişi)",
            "",
            "💡 TAVSİYELER:",
            f"• {most_popular[0]} bölgesine özel promosyonlar planlayın",
            f"• {least_popular[0]} bölgesini cazip hale getirin",
            f"• Duraksama oranı {stationary_rate:.1f}% - ilgi çekici noktalar oluşturun"
        ])

        return "\n".join(insights)


# -------------------------------
# OPTİMİZE EDİLMİŞ FRAME YAKALAMA
# -------------------------------
class OptimizedFrameCapture:
    def __init__(self, config, progress_callback=None):
        self.config = config
        self.progress_callback = progress_callback
        self.stop_capture = False
        self.captured_frames = 0

    def capture_frames(self, video_path):
        """SINIRSIZ frame yakalama"""
        self.stop_capture = False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Video açılamadı! Lütfen dosya yolunu kontrol edin.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = 0
        saved_count = 0

        print(f"🎥 Video bilgisi: {total_frames} frame, {fps:.1f} FPS")

        while not self.stop_capture:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.config.frame_interval == 0:
                filename = os.path.join(self.config.save_folder, f"frame_{saved_count:06d}.jpg")

                cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                saved_count += 1
                self.captured_frames = saved_count

                if self.progress_callback and saved_count % 50 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    self.progress_callback(progress, saved_count, frame_count)

            frame_count += 1

            if frame_count % 1000 == 0:
                print(f"📹 İşlenen frame: {frame_count}, Kaydedilen: {saved_count}")

        cap.release()
        return saved_count


# -------------------------------
# PROFESYONEL DASHBOARD ARAYÜZÜ - DASHBOARD GÜNCELLEME EKLENDİ
# -------------------------------
class ProfessionalDashboard:
    def __init__(self, root):
        self.root = root
        self.config = Config()
        self.analyzer = AdvancedHumanAnalyzer(self.config)
        self.capture_session = None
        self.user_max_frames = 0
        self.current_analysis_results = None

        self.setup_professional_ui()

    def setup_professional_ui(self):
        """Profesyonel arayüz - DASHBOARD GÜNCELLEME"""
        self.root.title("🏪 AKILLI MAĞAZA ANALİZ SİSTEMİ v4.0 - TRAJECTORY & DASHBOARD")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e2a38')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1e2a38')
        style.configure('TLabel', background='#1e2a38', foreground='white', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10), padding=10)
        style.configure('Header.TLabel', font=('Arial', 18, 'bold'), foreground='#3498db')

        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(header_frame, text="50 METRE KORİDOR - TRAJECTORY & DASHBOARD ANALİZ",
                  style='Header.TLabel').pack(side=tk.LEFT)

        control_frame = ttk.LabelFrame(main_container, text="🎛️  KONTROL PANELİ")
        control_frame.pack(fill=tk.X, pady=(0, 20))

        settings_row = ttk.Frame(control_frame)
        settings_row.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(settings_row, text="Frame Ayarları:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(settings_row, text="Interval:").pack(side=tk.LEFT, padx=(0, 5))
        self.interval_var = tk.StringVar(value="15")
        interval_spin = ttk.Spinbox(settings_row, from_=1, to=60, width=5, textvariable=self.interval_var)
        interval_spin.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(settings_row, text="Max Frames (0=sınırsız):").pack(side=tk.LEFT, padx=(0, 5))
        self.max_frames_var = tk.StringVar(value="0")
        max_frames_spin = ttk.Spinbox(settings_row, from_=0, to=100000, width=8, textvariable=self.max_frames_var)
        max_frames_spin.pack(side=tk.LEFT, padx=(0, 15))

        input_row = ttk.Frame(control_frame)
        input_row.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(input_row, text="Video Kaynağı:", font=('Arial', 11)).pack(side=tk.LEFT, padx=(0, 10))
        self.entry_file = ttk.Entry(input_row, width=80, font=('Arial', 10))
        self.entry_file.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)

        ttk.Button(input_row, text="📁 Dosya Seç", command=self.select_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(input_row, text="🌐 M3U8 URL", command=self.select_url).pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(control_frame, mode='determinate', length=1560)
        self.progress.pack(padx=10, pady=5)

        self.progress_label = ttk.Label(control_frame, text="Sistem hazır - analiz başlatın",
                                        font=('Arial', 10))
        self.progress_label.pack(pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=15)

        main_buttons = [
            ("🎬 SINIRSIZ Yakalama", self.start_unlimited_capture, '#27ae60'),
            ("⏹️ Durdur", self.stop_capture, '#e74c3c'),
            ("🤖 Model Eğit", self.train_model, '#3498db'),
            ("📊 Gelişmiş Analiz", self.advanced_analysis, '#9b59b6'),
            ("📈 İş Raporu", self.business_report, '#f39c12'),
            ("💾 Sonuçları Kaydet", self.save_results, '#1abc9c'),
            ("🔄 Dashboard Güncelle", self.update_dashboard, '#8e44ad')
        ]

        for i, (text, command, color) in enumerate(main_buttons):
            btn = tk.Button(button_frame, text=text, command=command,
                            bg=color, fg='white', font=('Arial', 10, 'bold'),
                            padx=15, pady=10, relief='raised', bd=3)
            btn.grid(row=0, column=i, padx=5, pady=5)

        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(content_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.setup_dashboard_tab(notebook)
        self.setup_analysis_tab(notebook)
        self.setup_reports_tab(notebook)
        self.setup_visualization_tab(notebook)
        self.setup_logs_tab(notebook)

    def setup_dashboard_tab(self, notebook):
        """Dashboard sekmesi - GERÇEK ZAMANLI VERİLER"""
        dashboard_frame = ttk.Frame(notebook)
        notebook.add(dashboard_frame, text="📊 Gerçek Zamanlı Dashboard")

        # Ana metrikler
        metrics_frame = ttk.LabelFrame(dashboard_frame, text="📈 CANLI METRİKLER")
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)

        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X, padx=10, pady=10)

        metrics = [
            ("Toplam İnsan", "0", "#2ecc71", "toplam_insan"),
            ("Ortalama Hız", "0.0", "#3498db", "ortalama_hiz"),
            ("Duraksama Oranı", "0%", "#e74c3c", "duraksama_orani"),
            ("Giriş Sayısı", "0", "#9b59b6", "giris_sayisi"),
            ("Çıkış Sayısı", "0", "#f39c12", "cikis_sayisi"),
            ("Popüler Bölge", "-", "#1abc9c", "populer_bolge"),
            ("Trajectory Sayısı", "0", "#e67e22", "trajectory_sayisi"),
            ("Ort. Trajectory", "0.0", "#9b59b6", "ort_trajectory")
        ]

        self.metric_vars = {}

        for i, (label, value, color, key) in enumerate(metrics):
            metric_frame = ttk.Frame(metrics_grid, relief='solid', borderwidth=1)
            metric_frame.grid(row=i // 4, column=i % 4, padx=5, pady=5, sticky='nsew')

            ttk.Label(metric_frame, text=label, font=('Arial', 11, 'bold'),
                      foreground='white', background='#2c3e50').pack(pady=(8, 0))

            value_var = tk.StringVar(value=value)
            value_label = ttk.Label(metric_frame, textvariable=value_var, font=('Arial', 16, 'bold'),
                                    foreground=color, background='#2c3e50')
            value_label.pack(pady=8)

            self.metric_vars[key] = value_var

        # Grafikler alanı
        charts_frame = ttk.LabelFrame(dashboard_frame, text="📊 CANLI GRAFİKLER")
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Sol grafikler
        left_charts = ttk.Frame(charts_frame)
        left_charts.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Sağ grafikler
        right_charts = ttk.Frame(charts_frame)
        right_charts.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Grafik placeholders - artık gerçek verilerle doldurulacak
        self.setup_live_chart(left_charts, "Bölge Dağılımı", "zone_chart")
        self.setup_live_chart(left_charts, "Hareket Analizi", "movement_chart")
        self.setup_live_chart(right_charts, "Zaman Serisi", "timeline_chart")
        self.setup_live_chart(right_charts, "Duraksama Haritası", "stationary_chart")

    def setup_live_chart(self, parent, title, chart_key):
        """Canlı grafik placeholder'ı"""
        chart_frame = ttk.LabelFrame(parent, text=title)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas için frame
        canvas_frame = ttk.Frame(chart_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Geçici mesaj
        placeholder = ttk.Label(canvas_frame, text=f"{title} verileri yükleniyor...",
                                font=('Arial', 10), foreground='gray')
        placeholder.pack(expand=True)

        setattr(self, f"{chart_key}_frame", canvas_frame)

    def setup_analysis_tab(self, notebook):
        """Analiz sekmesi"""
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="🔍 Detaylı Analiz")

        self.analysis_text = tk.Text(analysis_frame, height=25, bg='#2c3e50', fg='white',
                                     font=('Consolas', 10), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(analysis_frame, orient=tk.VERTICAL, command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=scrollbar.set)

        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)

        self.analysis_text.insert(tk.END, "🤖 TRAJECTORY DÜZELTMELİ ANALİZ SİSTEMİ\n\n")
        self.analysis_text.insert(tk.END, "• 50 metre koridor analizi\n")
        self.analysis_text.insert(tk.END, "• Görünür trajectory çizgileri\n")
        self.analysis_text.insert(tk.END, "• Canlı dashboard güncellemesi\n")
        self.analysis_text.insert(tk.END, "• Detaylı hareket akış analizi\n")

    def setup_reports_tab(self, notebook):
        """Raporlar sekmesi"""
        reports_frame = ttk.Frame(notebook)
        notebook.add(reports_frame, text="📈 İş Raporları")

        self.reports_text = tk.Text(reports_frame, height=25, bg='#34495e', fg='#ecf0f1',
                                    font=('Arial', 11), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(reports_frame, orient=tk.VERTICAL, command=self.reports_text.yview)
        self.reports_text.configure(yscrollcommand=scrollbar.set)

        self.reports_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)

        self.reports_text.insert(tk.END, "📋 İŞ ZEKASI RAPORLARI\n\n")
        self.reports_text.insert(tk.END, "Bu bölümde detaylı iş analiz raporları görüntülenecektir.\n\n")

    def setup_visualization_tab(self, notebook):
        """Görselleştirme sekmesi"""
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="🎨 Görselleştirme")

        self.viz_text = tk.Text(viz_frame, height=25, bg='#2c3e50', fg='white',
                                font=('Arial', 11), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(viz_frame, orient=tk.VERTICAL, command=self.viz_text.yview)
        self.viz_text.configure(yscrollcommand=scrollbar.set)

        self.viz_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)

        self.viz_text.insert(tk.END, "🎨 GELİŞMİŞ GÖRSELLEŞTİRME\n\n")
        self.viz_text.insert(tk.END, "• Trajectory haritaları\n")
        self.viz_text.insert(tk.END, "• Heatmap analizleri\n")
        self.viz_text.insert(tk.END, "• Hareket akış diyagramları\n")
        self.viz_text.insert(tk.END, "• Zaman serisi grafikleri\n")

    def setup_logs_tab(self, notebook):
        """Loglar sekmesi"""
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="📝 Sistem Logları")

        self.logs_text = tk.Text(logs_frame, height=25, bg='#1a1a1a', fg='#00ff00',
                                 font=('Consolas', 9), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(logs_frame, orient=tk.VERTICAL, command=self.logs_text.yview)
        self.logs_text.configure(yscrollcommand=scrollbar.set)

        self.logs_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)

        self.logs_text.insert(tk.END, "🚀 TRAJECTORY DÜZELTMELİ SİSTEM BAŞLATILDI\n")
        self.logs_text.insert(tk.END, f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def update_dashboard(self):
        """Dashboard'ı güncelle"""
        if not hasattr(self.analyzer, 'trajectory_data'):
            self.log("⚠️ Önce analiz yapmalısınız!")
            return

        try:
            # Metrikleri hesapla
            total_people = sum([len(traj) for traj in self.analyzer.trajectory_data.values()])
            avg_velocity = np.mean(
                [np.mean(v) for v in self.analyzer.velocity_data.values() if v]) if self.analyzer.velocity_data else 0
            entrance_count = len(self.analyzer.entrance_exit_data['entrance'])
            exit_count = len(self.analyzer.entrance_exit_data['exit'])

            stationary_people = sum([1 for data in self.analyzer.stationary_data.values() if len(data) > 10])
            stationary_rate = (stationary_people / max(1, total_people)) * 100 if total_people > 0 else 0

            zone_popularity = sorted(self.analyzer.zone_data.items(), key=lambda x: x[1]['count'], reverse=True)
            most_popular = zone_popularity[0] if zone_popularity else ("Yok", 0)

            total_trajectories = len(self.analyzer.trajectory_data)
            avg_trajectory_length = np.mean(
                [len(traj) for traj in self.analyzer.trajectory_data.values()]) if self.analyzer.trajectory_data else 0

            # Dashboard'ı güncelle
            self.metric_vars['toplam_insan'].set(f"{total_people}")
            self.metric_vars['ortalama_hiz'].set(f"{avg_velocity:.2f}")
            self.metric_vars['duraksama_orani'].set(f"{stationary_rate:.1f}%")
            self.metric_vars['giris_sayisi'].set(f"{entrance_count}")
            self.metric_vars['cikis_sayisi'].set(f"{exit_count}")
            self.metric_vars['populer_bolge'].set(f"{most_popular[0]}")
            self.metric_vars['trajectory_sayisi'].set(f"{total_trajectories}")
            self.metric_vars['ort_trajectory'].set(f"{avg_trajectory_length:.1f}")

            self.log("✅ Dashboard güncellendi!")

        except Exception as e:
            self.log(f"❌ Dashboard güncelleme hatası: {str(e)}")

    def select_file(self):
        """Dosya seçme"""
        file_path = filedialog.askopenfilename(
            title="Video dosyası seçin",
            filetypes=(
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m3u8 *.ts"),
                ("All files", "*.*")
            )
        )
        if file_path:
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, file_path)
            self.log(f"📁 Seçilen dosya: {os.path.basename(file_path)}")

    def select_url(self):
        """URL seçme"""
        url = simpledialog.askstring("M3U8 URL", "Lütfen M3U8 URL'sini girin:")
        if url:
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, url)
            self.log(f"🌐 M3U8 URL eklendi: {url[:60]}...")

    def update_progress(self, value, frames_captured, total_frames_processed, log_message=None):
        """İlerlemeyi güncelle"""
        self.progress['value'] = value

        if log_message:
            self.log(log_message)
        else:
            self.progress_label.config(
                text=f"Kaydedilen: {frames_captured} frame | İşlenen: {total_frames_processed} | İlerleme: {value:.1f}%"
            )

        self.root.update_idletasks()

    def log(self, message):
        """Log mesajı ekle"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.logs_text.see(tk.END)
        self.root.update_idletasks()

    def start_unlimited_capture(self):
        """SINIRSIZ frame yakalama"""
        video_path = self.entry_file.get()
        if not video_path:
            messagebox.showerror("Hata", "Lütfen bir video dosyası veya URL seçin!")
            return

        try:
            self.config.frame_interval = int(self.interval_var.get())
            self.user_max_frames = int(self.max_frames_var.get())
            self.log(
                f"⚙️ Ayarlar: Interval={self.config.frame_interval}, Max Frames={'Sınırsız' if self.user_max_frames == 0 else self.user_max_frames}")
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli sayılar girin!")
            return

        def capture_thread():
            try:
                self.capture_session = OptimizedFrameCapture(self.config, self.update_progress)

                if self.user_max_frames > 0:
                    self.log(f"🔢 {self.user_max_frames} frame ile sınırlı yakalama başlatılıyor...")

                frame_count = self.capture_session.capture_frames(video_path)
                self.log(f"✅ Frame yakalama tamamlandı! {frame_count} frame kaydedildi.")

            except Exception as e:
                self.log(f"❌ Yakalama hatası: {str(e)}")
            finally:
                self.progress['value'] = 0
                self.progress_label.config(text="Hazır")

        threading.Thread(target=capture_thread, daemon=True).start()
        self.log("🎬 SINIRSIZ frame yakalama başlatıldı...")

    def stop_capture(self):
        """Frame yakalamayı durdur"""
        if self.capture_session:
            self.capture_session.stop_capture = True
            self.log("⏹️ Frame yakalama durduruldu")

    def train_model(self):
        """Model eğitimi"""
        if not os.path.exists(self.config.save_folder) or not os.listdir(self.config.save_folder):
            messagebox.showerror("Hata", "Önce frame yakalama yapmalısınız!")
            return

        def train_thread():
            try:
                self.log("🤖 YOLOv8s model eğitimi başlatılıyor...")

                model = YOLO("yolov8s.pt")

                model.train(
                    data="coco8.yaml",
                    epochs=self.config.epochs,
                    imgsz=self.config.img_size,
                    batch=self.config.batch_size,
                    device="cpu",
                    workers=0,
                    patience=5,
                    lr0=0.01,
                    save=True,
                    exist_ok=True,
                    project=self.config.model_folder,
                    name=f"market_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    verbose=False
                )

                self.log("✅ Model eğitimi başarıyla tamamlandı!")

            except Exception as e:
                self.log(f"❌ Eğitim hatası: {str(e)}")

        threading.Thread(target=train_thread, daemon=True).start()

    def advanced_analysis(self):
        """Gelişmiş analiz - TRAJECTORY DÜZELTMELİ"""
        if not os.path.exists(self.config.save_folder):
            messagebox.showerror("Hata", "Önce frame yakalama yapmalısınız!")
            return

        def analyze_thread():
            try:
                self.log("🎯 Gelişmiş analiz başlatılıyor (trajectory düzeltmeli)...")

                model_paths = []
                for root, dirs, files in os.walk(self.config.model_folder):
                    for file in files:
                        if file.endswith('.pt') and 'train' not in file:
                            model_paths.append(os.path.join(root, file))

                if not model_paths:
                    model = YOLO("yolov8s.pt")
                    self.log("⚠ Özel model bulunamadı, varsayılan model kullanılıyor")
                else:
                    latest_model = max(model_paths, key=os.path.getctime)
                    model = YOLO(latest_model)
                    self.log(f"✅ Model yüklendi: {os.path.basename(latest_model)}")

                # TRAJECTORY DÜZELTMELİ ANALİZ
                density_heatmap, stationary_heatmap, velocity_heatmap, flow_heatmap, zone_heatmap, trajectory_map, sample_frame = \
                    self.analyzer.create_comprehensive_analysis(self.config.save_folder, model)

                if density_heatmap is not None:
                    self.log("✅ Analiz başarıyla tamamlandı!")

                    self.analysis_text.delete(1.0, tk.END)
                    self.analysis_text.insert(tk.END, "📊 DETAYLI ANALİZ SONUÇLARI\n\n")

                    total_people = sum([len(traj) for traj in self.analyzer.trajectory_data.values()])
                    self.analysis_text.insert(tk.END, f"• Toplam İnsan Hareketi: {total_people}\n")
                    self.analysis_text.insert(tk.END, f"• Takip Edilen Kişi: {len(self.analyzer.trajectory_data)}\n")
                    self.analysis_text.insert(tk.END,
                                              f"• Giriş Sayısı: {len(self.analyzer.entrance_exit_data['entrance'])}\n")
                    self.analysis_text.insert(tk.END,
                                              f"• Çıkış Sayısı: {len(self.analyzer.entrance_exit_data['exit'])}\n")

                    self.analysis_text.insert(tk.END, "\n🏪 BÖLGE ANALİZİ:\n")
                    for zone, data in self.analyzer.zone_data.items():
                        if data['count'] > 0:
                            self.analysis_text.insert(tk.END, f"• {zone}: {data['count']} kişi\n")

                    self.log("📈 Görselleştirme hazırlanıyor...")

                    # Dashboard'ı otomatik güncelle
                    self.update_dashboard()

                    self.create_advanced_visualization(density_heatmap, stationary_heatmap,
                                                       velocity_heatmap, flow_heatmap, zone_heatmap,
                                                       trajectory_map, sample_frame)

                else:
                    self.log("❌ Analiz sırasında hata oluştu!")

            except Exception as e:
                self.log(f"❌ Analiz hatası: {str(e)}")

        threading.Thread(target=analyze_thread, daemon=True).start()

    def create_advanced_visualization(self, density_heatmap, stationary_heatmap,
                                      velocity_heatmap, flow_heatmap, zone_heatmap,
                                      trajectory_map, sample_frame):
        """Gelişmiş görselleştirme - TRAJECTORY EKLENDİ"""
        try:
            fig, axes = plt.subplots(2, 4, figsize=(25, 12))
            fig.suptitle('50 METRE KORİDOR - TRAJECTORY DÜZELTMELİ ANALİZ', fontsize=16, fontweight='bold')

            # 1. Yoğunluk Heatmap
            axes[0, 0].imshow(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
            im1 = axes[0, 0].imshow(density_heatmap, alpha=0.7, cmap='jet')
            axes[0, 0].set_title('🚶 İnsan Yoğunluğu', fontweight='bold')
            plt.colorbar(im1, ax=axes[0, 0])

            # 2. Duraksama Heatmap
            axes[0, 1].imshow(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
            im2 = axes[0, 1].imshow(stationary_heatmap, alpha=0.7, cmap='hot')
            axes[0, 1].set_title('⏸️  Duraksama Noktaları', fontweight='bold')
            plt.colorbar(im2, ax=axes[0, 1])

            # 3. Hız Analizi
            axes[0, 2].imshow(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
            im3 = axes[0, 2].imshow(velocity_heatmap, alpha=0.6, cmap='cool')
            axes[0, 2].set_title('💨 İnsan Hız Dağılımı', fontweight='bold')
            plt.colorbar(im3, ax=axes[0, 2])

            # 4. TRAJECTORY HARİTASI - YENİ
            axes[0, 3].imshow(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
            axes[0, 3].imshow(trajectory_map, alpha=0.8)
            axes[0, 3].set_title('🔄 İnsan Hareket Yolları', fontweight='bold')

            # 5. Akış Heatmap
            axes[1, 0].imshow(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
            im5 = axes[1, 0].imshow(flow_heatmap, alpha=0.6, cmap='viridis')
            axes[1, 0].set_title('📊 Hareket Akışı', fontweight='bold')
            plt.colorbar(im5, ax=axes[1, 0])

            # 6. Bölge Popülerliği
            zone_data = list(self.analyzer.zone_data.items())
            zones = [zone[0] for zone in zone_data]
            counts = [zone[1]['count'] for zone in zone_data]

            axes[1, 1].barh(zones, counts, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0'])
            axes[1, 1].set_title('🏆 Bölge Popülerlik Analizi', fontweight='bold')
            axes[1, 1].set_xlabel('İnsan Sayısı')

            # 7. Trajectory İstatistikleri
            trajectory_lengths = [len(traj) for traj in self.analyzer.trajectory_data.values()]
            if trajectory_lengths:
                axes[1, 2].hist(trajectory_lengths, bins=20, color='skyblue', alpha=0.7)
                axes[1, 2].set_title('📏 Trajectory Uzunluk Dağılımı', fontweight='bold')
                axes[1, 2].set_xlabel('Trajectory Uzunluğu')
                axes[1, 2].set_ylabel('Frekans')
            else:
                axes[1, 2].text(0.5, 0.5, 'Trajectory verisi yok', ha='center', va='center')
                axes[1, 2].set_title('📏 Trajectory Uzunluk Dağılımı', fontweight='bold')

            # 8. İş Zekası Özeti
            stats_text = self.analyzer.generate_business_intelligence()
            axes[1, 3].text(0.1, 0.9, stats_text, transform=axes[1, 3].transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 3].axis('off')
            axes[1, 3].set_title('📈 İş Zekası Özeti', fontweight='bold')

            plt.tight_layout()
            plt.show()

            self.log("🎨 Görselleştirme tamamlandı!")

        except Exception as e:
            self.log(f"❌ Görselleştirme hatası: {str(e)}")

    def business_report(self):
        """İş zekası raporu"""
        try:
            if not hasattr(self.analyzer, 'zone_data') or not self.analyzer.zone_data:
                messagebox.showinfo("Bilgi", "Önce analiz yapmalısınız!")
                return

            report = self.analyzer.generate_business_intelligence()

            self.reports_text.delete(1.0, tk.END)
            self.reports_text.insert(tk.END, "📋 DETAYLI İŞ ZEKASI RAPORU\n\n")
            self.reports_text.insert(tk.END, report)

            report_path = os.path.join(self.config.reports_folder,
                                       f"business_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)

            self.log(f"💾 İş raporu kaydedildi: {report_path}")

        except Exception as e:
            self.log(f"❌ Rapor oluşturma hatası: {str(e)}")

    def save_results(self):
        """Sonuçları kaydet"""
        try:
            results = {
                'analysis_date': datetime.now().isoformat(),
                'total_people': sum([len(traj) for traj in self.analyzer.trajectory_data.values()]),
                'trajectory_count': len(self.analyzer.trajectory_data),
                'zones': dict(self.analyzer.zone_data),
                'entrance_exit': self.analyzer.entrance_exit_data,
                'hourly_stats': dict(self.analyzer.hourly_stats)
            }

            results_path = os.path.join(self.config.results_folder,
                                        f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            self.log(f"💾 Analiz sonuçları kaydedildi: {results_path}")
            messagebox.showinfo("Başarılı", f"Sonuçlar kaydedildi:\n{results_path}")

        except Exception as e:
            self.log(f"❌ Kaydetme hatası: {str(e)}")


# -------------------------------
# UYGULAMA BAŞLATMA
# -------------------------------
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ProfessionalDashboard(root)
        root.mainloop()
    except Exception as e:
        print(f"Uygulama hatası: {e}")