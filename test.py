import socket
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from threading import Thread
import time

# Default Settings
UDP_IP = "0.0.0.0"  # 모든 IP에서 수신
UDP_PORT = 5005     # 사용할 포트
BUFFER_SIZE = 2048  # 수신 버퍼 크기

NUM_CHIRPS = 64     # Chirp 수 (64 Chirps)
NUM_SAMPLES = 256   # Chirp 당 샘플 수 (256 Samples)
NUM_RECEIVERS = 3   # 수신 안테나 수 (Rx1, Rx2, Rx3)
SAMPLING_RATE = 1e6  # 1 MHz 샘플링
CARRIER_FREQ = 60e9  # 60 GHz carrier frequency
SPEED_OF_LIGHT = 3e8

# 수신 데이터를 저장할 버퍼
raw_data_buffer = np.zeros((NUM_RECEIVERS, NUM_CHIRPS, NUM_SAMPLES), dtype=np.complex64)
udp_status = "Disconnected"

# Generate Random Data
use_random_data = True  # Flag to enable random data generation

def generate_random_data():
    global raw_data_buffer
    while use_random_data:
        raw_data_buffer[:, :, :] = (
            np.random.randn(NUM_RECEIVERS, NUM_CHIRPS, NUM_SAMPLES) + 
            1j * np.random.randn(NUM_RECEIVERS, NUM_CHIRPS, NUM_SAMPLES)
        )
        time.sleep(0.1)  # Generate data every 100ms

# UDP Server
def udp_server():
    global raw_data_buffer, udp_status
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    udp_status = "Connected"
    while True:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            iq_data = np.frombuffer(data, dtype=np.complex64)
            iq_data = iq_data[:NUM_SAMPLES * NUM_RECEIVERS].reshape(NUM_RECEIVERS, NUM_SAMPLES)
            raw_data_buffer[:, :-1] = raw_data_buffer[:, 1:]
            raw_data_buffer[:, -1] = iq_data
        except BlockingIOError:
            pass

def process_data(raw_data, selected_receiver):
    data = raw_data[selected_receiver]
    time_domain = np.abs(data[-1]) / (np.max(np.abs(data)) + 1e-6)
    range_fft = np.fft.fft(data, axis=1)
    range_fft = np.abs(range_fft[:, :NUM_SAMPLES // 2])
    range_fft = 20 * np.log10(range_fft + 1e-6)
    range_res = SPEED_OF_LIGHT / (2 * SAMPLING_RATE)
    range_axis = np.arange(NUM_SAMPLES // 2) * range_res
    doppler_fft = np.fft.fft(data, axis=0)
    doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
    doppler_map = np.abs(doppler_fft[:, :NUM_SAMPLES // 2])
    doppler_map = 20 * np.log10(doppler_map + 1e-6)
    return time_domain, range_fft[-1], range_axis, doppler_map

# PyQtGraph Application
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
win.setWindowTitle("Radar Data Viewer")
win.resize(1200, 900)

# Main Layout
main_layout = QtWidgets.QVBoxLayout()
win.setLayout(main_layout)

# Top Layout
settings_receiver_split_layout = QtWidgets.QHBoxLayout()
main_layout.addLayout(settings_receiver_split_layout)

# Left Top Layout (Settings)
settings_layout = QtWidgets.QFormLayout()
settings_group = QtWidgets.QGroupBox("Settings")
settings_group.setLayout(settings_layout)
settings_receiver_split_layout.addWidget(settings_group)

udp_ip_input = QtWidgets.QLineEdit(UDP_IP)
udp_port_input = QtWidgets.QLineEdit(str(UDP_PORT))
buffer_size_input = QtWidgets.QLineEdit(str(BUFFER_SIZE))
num_chirps_input = QtWidgets.QLineEdit(str(NUM_CHIRPS))
num_samples_input = QtWidgets.QLineEdit(str(NUM_SAMPLES))
num_receivers_input = QtWidgets.QLineEdit(str(NUM_RECEIVERS))
sampling_rate_input = QtWidgets.QLineEdit(str(SAMPLING_RATE))
carrier_freq_input = QtWidgets.QLineEdit(str(CARRIER_FREQ))

settings_layout.addRow("UDP IP:", udp_ip_input)
settings_layout.addRow("UDP Port:", udp_port_input)
settings_layout.addRow("Buffer Size:", buffer_size_input)
settings_layout.addRow("Num Chirps:", num_chirps_input)
settings_layout.addRow("Num Samples:", num_samples_input)
settings_layout.addRow("Num Receivers:", num_receivers_input)
settings_layout.addRow("Sampling Rate:", sampling_rate_input)
settings_layout.addRow("Carrier Frequency:", carrier_freq_input)

apply_button = QtWidgets.QPushButton("Apply Settings")
def apply_settings():
    global UDP_IP, UDP_PORT, BUFFER_SIZE, NUM_CHIRPS, NUM_SAMPLES, NUM_RECEIVERS, SAMPLING_RATE, CARRIER_FREQ
    UDP_IP = udp_ip_input.text()
    UDP_PORT = int(udp_port_input.text())
    BUFFER_SIZE = int(buffer_size_input.text())
    NUM_CHIRPS = int(num_chirps_input.text())
    NUM_SAMPLES = int(num_samples_input.text())
    NUM_RECEIVERS = int(num_receivers_input.text())
    SAMPLING_RATE = float(sampling_rate_input.text())
    CARRIER_FREQ = float(carrier_freq_input.text())
    print("Settings Applied")
apply_button.clicked.connect(apply_settings)
settings_layout.addWidget(apply_button)

# Right Top Layout (Receiver Selection)
receiver_layout = QtWidgets.QVBoxLayout()
receiver_group = QtWidgets.QGroupBox("Receiver Selection")
receiver_group.setLayout(receiver_layout)
settings_receiver_split_layout.addWidget(receiver_group)

selected_receiver = 0

def set_receiver(rx):
    global selected_receiver
    selected_receiver = rx
    time_plot.setTitle(f"Time Domain (Rx{rx + 1})")
    range_plot.setTitle(f"Range Spectrum (Rx{rx + 1})")
    doppler_plot.setTitle(f"Range-Doppler Map (Rx{rx + 1})")
    print(f"Switched to Rx{rx + 1}")

rx1_button = QtWidgets.QPushButton("Rx1")
rx1_button.clicked.connect(lambda: set_receiver(0))
receiver_layout.addWidget(rx1_button)

rx2_button = QtWidgets.QPushButton("Rx2")
rx2_button.clicked.connect(lambda: set_receiver(1))
receiver_layout.addWidget(rx2_button)

rx3_button = QtWidgets.QPushButton("Rx3")
rx3_button.clicked.connect(lambda: set_receiver(2))
receiver_layout.addWidget(rx3_button)

# Bottom Layout (Plots)
plot_layout = QtWidgets.QHBoxLayout()
main_layout.addLayout(plot_layout)

plot_widget = pg.GraphicsLayoutWidget()
plot_layout.addWidget(plot_widget)

# Plots
time_plot = plot_widget.addPlot(title="Time Domain (Rx1)")
time_curve = time_plot.plot(pen="y")
time_plot.setLabel("bottom", "Samples", "Total Samples")
time_plot.setLabel("left", "ADC Amp")
time_plot.showGrid(x=True, y=True)
plot_widget.nextRow()
range_plot = plot_widget.addPlot(title="Range Spectrum (Rx1)")
range_curve = range_plot.plot(pen="b")
range_plot.setLabel("bottom", "Range [m]")
range_plot.setLabel("left", "Magnitude [dBFS]")
range_plot.showGrid(x=True, y=True)
plot_widget.nextRow()
doppler_plot = plot_widget.addPlot(title="Range-Doppler Map (Rx1)")
doppler_image = pg.ImageItem()
doppler_plot.addItem(doppler_image)
doppler_plot.setLabel("bottom", "Velocity [m/s]")
doppler_plot.setLabel("left", "Range [m]")
doppler_plot.showGrid(x=True, y=True)

# Define colormap
colormap = pg.colormap.getFromMatplotlib("viridis")
doppler_colorbar = pg.ColorBarItem(values=(0, 100), colorMap=colormap, label="Magnitude (dB)")
doppler_colorbar.setImageItem(doppler_image, insert_in=doppler_plot)

def update_plots():
    while True:
        time_domain, range_fft, range_axis, doppler_map = process_data(raw_data_buffer, selected_receiver)

        # Update Time Domain
        total_samples = len(time_domain) * NUM_CHIRPS
        time_curve.setData(np.arange(total_samples), np.tile(time_domain, NUM_CHIRPS))

        # Update Range Spectrum
        range_curve.setData(range_axis, range_fft)

        # Update Range-Doppler Map
        doppler_image.setImage(doppler_map.T, autoLevels=True)

        time.sleep(0.1)  # Refresh every 100ms

if __name__ == "__main__":
    if use_random_data:
        random_data_thread = Thread(target=generate_random_data, daemon=True)
        random_data_thread.start()

    udp_thread = Thread(target=udp_server, daemon=True)
    udp_thread.start()

    plot_thread = Thread(target=update_plots, daemon=True)
    plot_thread.start()

    win.show()
    QtWidgets.QApplication.instance().exec_()
